import torch
from safetensors.torch import load_file

# Replace this path with the path to your .safetensors file
safetensors_path = "/home/tangzichen/ChatMed/output/chatmed-llama-7b-pt-v0/adapter_model.safetensors"
# This will be the output path for the .bin file
bin_path = "/home/tangzichen/ChatMed/output/chatmed-llama-7b-pt-v0/adapter_model.bin"

# Load the .safetensors file
data = load_file(safetensors_path)

# Save the data to a .bin file
torch.save(data, bin_path)

print(f"Converted {safetensors_path} to {bin_path}")