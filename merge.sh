python src/chatmed_llama_peft/merge_llama_with_chinese_lora.py \
    --base_model /home/tangzichen/ChatMed/resources/Llama-2-7b-hf \
    --lora_model /home/tangzichen/ChatMed/resources/chinese_llama_plus_lora_7b,/home/tangzichen/ChatMed/resources/chinese_alpaca_plus_lora_7b \
    --output_type huggingface \
    --output_dir ./resources/chinese-llama-alpaca-plus-lora-7b
