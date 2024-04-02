from datasets import load_dataset

dataset = load_dataset("michaelwzhu/ShenNong_TCM_Dataset")
dataset.save_to_disk('/home/tangzichen/ChatMed/dataset')  # Replace with your path
