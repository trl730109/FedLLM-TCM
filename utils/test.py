import numpy as np
import os
import torch
import peft
import transformers
import sys

'''
conda_site_packages_path = "/home/tangzichen/miniconda3/envs/new_tcm/lib/python3.9/site-packages"
if conda_site_packages_path not in sys.path:
    sys.path.insert(0, conda_site_packages_path)
'''
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
'''
#import  sys
#sys.path.append("./")
conda_site_packages_path = "/home/tangzichen/miniconda3/envs/new_tcm/lib/python3.9/site-packages"
print("sys", sys.path)
# if conda_site_packages_path not in sys.path:
sys.path.insert(0, conda_site_packages_path)
print("sysII", sys.path)
'''

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

os.environ["WANDB_MODE"] = "disable"

def generate_text(prompt, tokenizer, model, max_length=50, num_return_sequences=1):
    """
    Generate text based on the given prompt.

    Parameters:
    - prompt: The text prompt to generate text from.
    - tokenizer: The tokenizer for encoding the input prompt.
    - model: The model used for generating text.
    - max_length: The maximum length of the generated text.
    - num_return_sequences: The number of generated text sequences to return.

    Returns:
    - generated_texts: A list of generated text sequences.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
        )
    
    generated_texts = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
    return generated_texts

if __name__ == "__main__":
    #print(peft.__file__)
    #print(sys.path)
    lora_model_dir = "/home/tangzichen/ChatMed/output/chatmed-llama-7b-pt-v1"
    base_model_path = "/home/tangzichen/ChatMed/resources/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    base_model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
        )

    lora_model = PeftModel.from_pretrained(
                base_model,
                lora_model_dir,
                device_map={"": "cuda:0"},
                torch_dtype=torch.float16,
            )
    print("Lora model loaded.")
    print(f"Merging with merge_and_unload...")
    base_model = lora_model.merge_and_unload()
    #model = PeftModel.from_pretrained(base_model, lora_model_dir)
    print("Successfully load the fine-tuned model!")
    
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'quit':
            break
        generated_text = generate_text(prompt, tokenizer, base_model)
        print("Generated text:", generated_text[0])
    print("Model loaded. Enter 'quit' to exit.")
