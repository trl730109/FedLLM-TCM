from transformers import AutoModel

# Specify the model name with LoRA weights from Hugging Face
model_name = "your_model_name_with_lora"  # Replace this with the actual model name

# Load the model from Hugging Face
model = AutoModel.from_pretrained(model_name)

# Specify the local path where you want to save the model
local_save_path = "./local_model_directory"

# Save the model and its configuration locally
model.save_pretrained(local_save_path)

print(f"Model saved locally at {local_save_path}")
