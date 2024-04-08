import logging

import jieba
import numpy as np
import math
import os
import sys
import datetime
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List, Dict, Any, Mapping
from pathlib import Path
import datasets
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from fl_utils import *


import transformers
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge

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

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict
from ChatMed.src.FedLLM.fl_utils import train
from config_classes import ModelArguments, DataTrainingArguments, MyTrainingArguments, FLTrainingArguments

os.environ["WANDB_MODE"] = "disable"

Categories = {
    "Respiratory System Diseases": ["喘", "呼吸", "肺","鼻","气管","咽喉","咳嗽"],  #呼吸系统疾病
    "Digestive System Diseases": ["胃", "肠","腹","口","咽","喉","粪便","便"], #消化系统疾病
    "Cardiovascular Diseases": ["经络", "心", "中风","血","脉","头"], #心血管疾病
    "Musculoskeletal Disorders": ["筋", "骨", "髓","风湿"], #肌肉骨骼疾病
    #"Neurological Disorders": ["阿尔茨海默病", "帕金森病", "多发性硬化症"], #神经系统疾病
    "Endocrine Disorders": ["糖尿病", "甲状腺疾病", "肾上腺功能减退"], #内分泌失调
    "Kidney and Urinary Diseases": ["肾", "尿", "精","虚"], #肾脏和泌尿系统疾病
    #"Liver and Gallbladder Disorders": ["肝炎", "肝硬化", "胆结石"], #肝脏和胆囊疾病
    "Skin Diseases": ["疹", "癣", "疮","痘","面","肿"], #皮肤病
    #"Mental and Emotional Disorders": ["抑郁", "焦虑", "神"], #精神和情感障碍
    "Traditional chinese medicine": ["功效","作用","用法","治疗","推荐","文献"],
    "Others": ["其他疾病"] #其他疾病
    }

def convert_index_to_location(index):
    categories = [
    "Respiratory System Diseases",
    "Digestive System Diseases",
    "Cardiovascular Diseases",
    "Musculoskeletal Disorders",
    "Endocrine Disorders",
    "Kidney and Urinary Diseases",
    "Skin Diseases",
    "Traditional Chinese Medicine",
    "Others"
    ]
    return categories[index]


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments, FLTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, fl_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, fl_args = parser.parse_args_into_dataclasses()
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],)


    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    #Data loading
    data_path = "/home/tangzichen/ChatMed/dataset"
    dataset = load_from_disk(data_path)
    raw_dataset = dataset['train']
    num_clients = fl_args.num_clients
    output_type = 'huggingface'

    partition_datasets = partition_dataset(raw_dataset, partition_criteria=fl_args.data_partition, num_partitions=num_clients, local_directory=None,concentration=0.5, categories=Categories)
    
    for round in range(fl_args.num_rounds):
        logger.info(f"Starting FL Round {round+1}/{fl_args.num_rounds}")

        base_model = LlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    load_in_8bit=False,
                    torch_dtype=torch.float16,
                    device_map={"": "cpu"},
                )

        arr = np.arange(num_clients)
        np.random.shuffle(arr)
        selected = arr[:int(num_clients * fl_args.sample)]
        peft_model_dict = {}
        for client_id in selected:
            client_str = f"client_{client_id}"
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./output/FedLLM/{client_str}"
            peft_model_dict[client_id] = output_dir

            if (fl_args.data_partition == "quantity_skew"):               
                #client_str = f"client_{client_id}"
                local_dataset = partition_datasets[client_str]
            elif (fl_args.data_partition == "disease_category"):
                assert num_clients < 9, "Currently only support 9 kinds of diseases."
                local_dataset = partition_datasets[convert_index_to_location(client_id)]

            train(client_id, local_dataset, model_args, data_args, training_args, fl_args, output_dir)

        updated_model = model_merging(model_args.model_name_or_path, peft_model_dict, len(selected))
        base_model.load_state_dict(updated_model)
        logger.info(f"Finished training {len(selected)} clients")
        
        if output_type=='huggingface':
            print("Saving to Hugging Face format...")
            LlamaForCausalLM.save_pretrained(base_model, output_dir) #, state_dict=deloreanized_sd)
            '''
        else: # output_type=='pth
            print("Saving to pth format...")

            base_model_sd = base_model.state_dict()
            del lora_model, base_model, lora_model_sd

            params = params_of_models[model_size]
            num_shards = num_shards_of_models[model_size]
            n_layers = params["n_layers"]
            n_heads = params["n_heads"]
            dim = params["dim"]
            dims_per_head = dim // n_heads
            base = 10000.0
            inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

            save_shards(model_sd=base_model_sd, num_shards=num_shards)
        '''
