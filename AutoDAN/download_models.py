import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


model_name = "lmsys/vicuna-7b-v1.5"
base_model_path ="./models/llama2/vicuna-7b-v1.5"

if not os.path.exists(base_model_path):
    os.makedirs(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True, use_cache=False)
#Save the model and the tokenizer
model.save_pretrained(base_model_path, from_pt=True)
tokenizer.save_pretrained(base_model_path, from_pt=True)
