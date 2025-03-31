from us_helpers import delete_specific_directories, find_checkpoint
delete_specific_directories()
import sys
if len(sys.argv) >= 2:
    chpnt = find_checkpoint(sys.argv[1])
else:
    chpnt = find_checkpoint()
import os
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
import colorama
from colorama import Fore, Back, Style
from peft.peft_model import PeftModelForCausalLM
from unsloth import UnslothTrainer, UnslothTrainingArguments
from finetune_sys_prompt import finetune_sys_prompt
import json
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
colorama.init()
model: str | nn.Module | PreTrainedModel = None

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= chpnt["path"], 
    max_seq_length=8000,
    dtype = None,
    load_in_4bit = True
)
# After from_pretrained, model is now <class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'> 
FastLanguageModel.for_inference(model)

print(f'{Fore.CYAN}After for_inference{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# After get_peft_model, model is now <class 'peft.peft_model.PeftModelForCausalLM'> 
tokenizer: LlamaTokenizerFast = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    map_eos_token = True
)
print(f'{Fore.CYAN}After get_chat_template{Style.RESET_ALL}, tokenizer is now{Fore.GREEN}', type(tokenizer), f"{Style.RESET_ALL}")
from transformers import TextStreamer
import re
pat = re.compile(r"<\|im_start\|>assistant((?:.|\n)*)<\|im_end\|>")
class TextCollector(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.text = ""
        
    def on_finalized_text(self, text, stream_end = False):
        self.text += text
        super().on_finalized_text(text, stream_end)
text_collector = TextCollector(tokenizer)
todays_notes_path = "D:\DocumentsHDD\ToCards\ToCards_03_21_25.txt" # logic to find today's notes goes here
todays_str = None
with open(todays_notes_path, 'r', encoding='utf-8') as f:
    todays_str = f.read()


t1 = "ch changes the dir in linux"
t2 = """AI

the number of examples is the.. number of examples

the batch size is.. how many examples to process at once

the iteration or learning step is..  a completed batch

An epoch is.. the completion of all batches. Full pass

–--

people use the word “vectorization” to mean “parallelization” a lot

--
(expression for outer_var in outer_iterable for inner_var in inner_iterable) 

same as 

result = [] 
for outer_var in outer_iterable:
 for inner_var in inner_iterable:
  result.append(expression)"""
t3 = """AI

the number of examples is the.. number of examples

the batch size is.. how many examples to process at once

the iteration or learning step is..  a completed batch

An epoch is.. the completion of all batches. Full pass"""
def prompt_model(inp):
    messages = [
            {"role": "system", "content": finetune_sys_prompt},
            {"role": "user", "content": inp}
        ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    not_sure_what_do_this = model.generate(input_ids = inputs, pad_token_id=tokenizer.eos_token_id, streamer = text_collector, use_cache = True)
    
if __name__ == "__main__":
    while True:
        text_collector.text = ""
        print("\nwhat would you like to say?\n")
        inp = ""
        while True:
            line = input("")
            if line == "/today" or "/t" or line == "":
                line = todays_str
                break
            elif line == "/t1":
                line = t1
                break
            elif line == "/t2":
                line = t2
                break
            elif line == "/t3":
                line = t3
                break
            if line[-2:] == "//":
                inp += line[:-2]
                break
            inp += line + '\n'
        
        prompt_model(inp)
        print("===========")
        
        m = re.search(pat, text_collector.text)
        if not m:
            raise Exception("couldn't find assistant output")
        try:
            loaded = json.loads(m[1])
            load_dump = json.dumps(loaded, indent=2, ensure_ascii=False)
            print("Load, output:")
            print(f"type: {type(loaded)},", load_dump)
            print()
        except Exception as e:
            print(sys.exc_info())
            print("could not load output as json")
            print()
            print(m[1])
        # we are not dead
