import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # No need to clear CUDA cache since we're using CPU
    model_dir = '/home/tangzichen/ChatMed/resources/chinese-llama-alpaca-plus-lora-7b'  # Change to your actual model directory path
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Explicitly set to use CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)  # Move model to the CPU

    prompt_text = "今天天气如何？"  # Change this with your Chinese text prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)  # Even though using CPU, keep this for consistency

    # Generate responses
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True,add_prefix_space=False)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
