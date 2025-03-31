import os
from us_helpers import find_latest_checkpoint, delete_specific_directories
delete_specific_directories()
import torch.version
from unsloth.save import create_ollama_modelfile
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import Trainer, TrainingArguments
import code
import traceback
from transformers import PreTrainedModel
import torch.nn as nn
import sys
import colorama
from colorama import Fore, Back, Style
from peft.peft_model import PeftModelForCausalLM
colorama.init()

dir_name = find_latest_checkpoint()
print(dir_name)
model: str | nn.Module | PreTrainedModel = None
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name=os.path.join(dir_name), 
#     max_seq_length=2048
# )
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ggufmodel", 
    max_seq_length=2048
)
print('After from pretrained, model is now', type(model))
# After from_pretrained, model is now <class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'> 
print(f'{Fore.CYAN}After from_pretrained{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# model = FastLanguageModel.get_peft_model(model)
print(f'{Fore.CYAN}After get_peft_model{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# After get_peft_model, model is now <class 'peft.peft_model.PeftModelForCausalLM'> 
"""
    So in my google collab example I think the template was supposed to change
    back to mistral at some point but I just never did it.

    there is a mistral option for chat_template. That might work better if things
    are weird? Maybe?vvvvvv
"""
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml", # <--Having to do this manually seems bad. Need to find out how to find it
    map_eos_token = True
)
print("tokenizer._ollama_modelfile", getattr(tokenizer, "_ollama_modelfile", "NOT FOUND"))
# model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method = "q4_k_m")
mf = create_ollama_modelfile(tokenizer, "ggufmodel")
if mf is not None:
        modelfile_location = os.path.join("ggufmodel", "Modelfile") # The 1st arg I had to put in myself, prob not good either
        with open(modelfile_location, "w") as file:
            file.write(mf)