#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging

import jieba
import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk, DatasetDict
from huggingface_hub import hf_hub_download

import transformers
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
from config_classes import ModelArguments, DataTrainingArguments, MyTrainingArguments

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from sklearn.metrics import accuracy_score

import  sys
sys.path.append("./")

import peft
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

os.environ["WANDB_MODE"] = "disable"


def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch

class GroupTextsBuilder:
    def __init__(self,max_seq_length):
        self.max_seq_length = max_seq_length
    def __call__(self, examples):
        # Concatenate all texts.
        firsts = {k:examples[k][0][0] for k in examples.keys()}
        lasts = {k:examples[k][0][-1] for k in examples.keys()}
        contents = {k:sum([vi[1:-1] for vi in v],[]) for k,v in examples.items()}
        total_length = len(contents[list(examples.keys())[0]])

        content_length = self.max_seq_length - 2
        if total_length >= content_length:
            total_length = (total_length // content_length ) * content_length
        # Split by chunks of max_len.
        result = {
            k: [ [firsts[k]] + t[i : i + content_length] + [lasts[k]] for i in range(0, total_length, content_length)] for k, t in contents.items()}
        return result


logger = logging.getLogger(__name__)


def train(client_id, local_dataset, model_args, data_args, training_args,output_dir, model):
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    

    # Detecting last checkpoint.
    last_checkpoint = None
    logger.info(f"checkpoint detecting is {os.path.isdir(output_dir)}")
    if os.path.isdir(output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    logger.info("Finish loading the tokenizer.")
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    prompt_column = "query"
    response_column = "response"
    def tokenize_function(examples):
        list_texts = []
        for i in range(len(examples[prompt_column])):
            query = examples[prompt_column][i]
            response = examples[response_column][i]
            prompt = "<s>问：\n{}\n答：\n{}\n</s>".format(query.strip(), response.strip())
            list_texts.append(
                prompt
            )
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(list_texts, add_special_tokens=False)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    logger.info("Finish processing the block-size.")
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    logger.info("Start processign the dataset processing.")
    #Do the data preprocessing here using the datasets mapping function which performs operations in parallel to each datd point
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        raw_datasets = local_dataset
        tokenized_dataset = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=["query", "response"],
                    load_from_cache_file=False,
                    keep_in_memory=False,
                    # cache_file_names = {k: os.path.join(data_args.dataset_cache_dir, f'tokenized.arrow') for k in raw_datasets},
                    desc="Running tokenizer on dataset",
                )
        print("tokenized_dataset: ", tokenized_dataset)
        logger.info("Tokenize the dataset.")
        grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    #cache_file_names = {k: os.path.join(data_args.dataset_cache_dir, f'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {block_size}",
                )
        # processed_dataset = grouped_datasets
        lm_datasets = grouped_datasets.train_test_split(test_size =0.003)
        logger.info("Finish processing the dataset.")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            hypothesis = ' '.join(hypothesis)
            if not hypothesis:
                hypothesis = "-"
            scores = rouge.get_scores(hypothesis, ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    if training_args.do_train:
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
        logger.info(tokenizer.decode(train_dataset[1]['input_ids']))
        logger.info(tokenizer.decode(train_dataset[2]['input_ids']))
        logger.info(tokenizer.decode(train_dataset[3]['input_ids']))
    if training_args.do_eval:
        eval_dataset = lm_datasets["test"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    # if model_args.model_name_or_path:
    #     torch_dtype = (
    #         model_args.torch_dtype
    #         if model_args.torch_dtype in ["auto", None]
    #         else getattr(torch, model_args.torch_dtype)
    #     )
    #     model = LlamaForCausalLM.from_pretrained(
    #         model_args.model_name_or_path,
    #         from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #         config=config,
    #         cache_dir=model_args.cache_dir,
    #         revision=model_args.model_revision,
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         torch_dtype=torch_dtype,
    #         # device_map="auto",
    #         # low_cpu_mem_usage=True
    #     ).half()
    # else:
    #     model = AutoModelForCausalLM.from_config(config)
    #     n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    #     logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    if training_args.peft_path is not None:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save.split(',') if training_args.modules_to_save.strip() else None
        lora_rank = training_args.lora_rank
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        print('Target modules are ', target_modules)
        print('Lora rank is ', lora_rank)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False, 
            r=lora_rank, lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout,
            modules_to_save=None)
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))
    for n, p in model.named_parameters():
        print(n, p.requires_grad)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        print("*" * 50)
        print("resume_from_checkpoint: ", checkpoint)
        print("*" * 50)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.model.save_pretrained(output_dir,safe_serialization=False)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def partition_dataset_with_quantity_skew(raw_dataset, num_of_clients, concentration):
    raw_dataset = raw_dataset.shuffle(seed=42)
    
    total_size = len(raw_dataset)
    partitions = DatasetDict()
    partition_proportions = np.random.dirichlet(alpha=[concentration] * num_of_clients)
    
    start_index = 0
    for i, proportion in enumerate(partition_proportions):
        num_samples = int(proportion * total_size)
        end_index = start_index + num_samples
        end_index = min(end_index, total_size)
        partition_dataset = raw_dataset.select(range(start_index, end_index))
        partition_name = f'client_{i+1}'
        partitions[partition_name] = partition_dataset
        
        start_index = end_index
    
    return partitions

def partition_dataset_with_location(raw_dataset, categories):
    partitions = {category: [] for category in categories.keys()}

    for i, record in enumerate(raw_dataset):
        query = record["query"]
        answer = record["response"]
        for category in categories:
            if category != "Others":
                if any(keyword in query or keyword in answer for keyword in categories[category]):
                    partitions[category].append(i)
                    break
            else:
                partitions["Others"].append(i)

    for category, indices in partitions.items():
        partition_dataset = raw_dataset.select(indices)
        partitions[category] = partition_dataset

    return partitions

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def partition_dataset(raw_dataset, partition_criteria='disease_category', num_partitions=10, concentration=0.5, categories=None):
    if (partition_criteria == "disease_category"):
        partitioned_datasets = partition_dataset_with_location(raw_dataset, categories)
    elif (partition_criteria == "quantity_skew"):
        partitioned_datasets = partition_dataset_with_quantity_skew(raw_dataset, num_partitions, concentration)
    
    return partitioned_datasets 


def model_merging(base_model, lora_model_path_dict, number_of_clients):
    '''
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    if base_model.get_input_embeddings().weight.size(0) != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))
        print(f"Extended vocabulary size to {len(tokenizer)}")
    '''
    first_weight = base_model.model.layers[0].self_attn.q_proj.weight
    first_weight_old = first_weight.clone()

    base_model_sd = base_model.state_dict()
    for key, value in lora_model_path_dict.items():
        print(f"Loading LoRA {value}")
        try:
            lora_model_sd = torch.load(os.path.join(value,'adapter_model.bin'),map_location='cpu')
        except FileNotFoundError:
            print("Cannot find lora model on the disk. Downloading lora model from hub...")
            filename = hf_hub_download(repo_id=value,filename='adapter_model.bin')
            lora_model_sd = torch.load(filename,map_location='cpu')

        lora_config = peft.LoraConfig.from_pretrained(value)
        lora_scaling = lora_config.lora_alpha / lora_config.r
        fan_in_fan_out = lora_config.fan_in_fan_out
        lora_keys = [k for k in lora_model_sd if 'lora_A' in k]
        non_lora_keys = [k for k in lora_model_sd if not 'lora_' in k]

        for k in non_lora_keys:
            print(f"merging {k}")
            original_k = k.replace('base_model.model.','')
            base_model_sd[original_k].copy_(lora_model_sd[k])

        for k in lora_keys:
            print(f"merging {k}")
            original_key = k.replace('.lora_A','').replace('base_model.model.','')
            assert original_key in base_model_sd
            lora_a_key = k
            lora_b_key = k.replace('lora_A','lora_B')
            base_model_sd[original_key] += (
                (transpose(lora_model_sd[lora_b_key].float() @ lora_model_sd[lora_a_key].float(),fan_in_fan_out) * lora_scaling) / number_of_clients
            )
            assert base_model_sd[original_key].dtype == torch.float16

            # did we do anything?
            assert not torch.allclose(first_weight_old, first_weight)
        print(f"Finish merging the client {key} from {value}")
    return base_model_sd
