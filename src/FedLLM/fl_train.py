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


@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj")
    lora_rank : Optional[int] = field(default=8)
    lora_dropout : Optional[float] = field(default=0.1)
    lora_alpha : Optional[float] = field(default=32.)
    modules_to_save : Optional[str] = field(default='embed_tokens,lm_head')
    debug_mode : Optional[bool] = field(default=False)
    peft_path : Optional[str] = field(default=None)
    leraning_rate : Optional[float] = field(default=1e-5)

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


    partition_datasets = partition_dataset(raw_dataset, partition_criteria="quantity_skew", num_partitions=10, local_directory=None,concentration=0.5, categories=None)
    
    for round in range(fl_args.num_rounds):
        logger.info(f"Starting FL Round {round+1}/{fl_args.num_rounds}")
        
        # Sample a fraction of clients to participate in this round if client_fraction < 1.0
        # selected_clients = sample_clients(num_clients=fl_args.num_clients, fraction=fl_args.client_fraction)
        arr = np.arange(fl_args.num_clients)
        np.random.shuffle(arr)
        selected = arr[:int(fl_args.num_clients * fl_args.sample)]
        peft_model_dict = {}
        for client_id in selected:
            client_str = f"client_{client_id}"
            local_dataset = partition_datasets[client_str]
            client_model_or_path = train(client_id, local_dataset, model_args, data_args, training_args)

            peft_model_dict[client_id] = get_peft_model_state_dict(client_model_or_path)
        logger.info(f"Finished training {len(selected)} clients")
        # Aggregate the updates collected from clients
        # Assume aggregate_updates is a function that implements FedAvg or other aggregation methods
        #TODO Aggregate the model updates and store it for next communication round
        aggregated_updates = aggregate_updates(peft_model_dict, aggregation_method=fl_args.aggregation_method)
        

        #TODO Merge the model updates to store in a specified path
        global_model = apply_aggregated_updates(global_model, aggregated_updates)
        
        if fl_args.eval_every_round:
            #TODO Perform model evaluation after each round or in specific rounds
            eval_metrics = evaluate_model(global_model)
            print(f"Evaluation metrics after round {round+1}: {eval_metrics}")
        
        # Optionally save the global model based on the save_strategy
        if fl_args.save_strategy == "all_rounds" or (fl_args.save_strategy == "end" and round == fl_args.num_rounds - 1):
            #TODO Save the global model to a specified path
            save_model(global_model, round)
