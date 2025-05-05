from us_helpers import delete_specific_directories, find_checkpoint
delete_specific_directories()
import sys
# chpnt_dir = "note_maker_before_src_training"
# chpnt_dir = "outputs/obj_to_src"
chpnt_dir = "outputs/note_maker"
chpnt = find_checkpoint(
    chpnt_num=sys.argv[1] if len(sys.argv) >= 2 else None, 
    chpnts_dir=chpnt_dir
)
from finetune_sys_prompt import *
# SYS_OBJ = None
SYS_PROMPT = finetune_sys_prompt
SYS_OBJ = {"role": "system","content": SYS_PROMPT}
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
import json
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from anki_csv_reader import *
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
# todays_notes_path = "D:\DocumentsHDD\ToCards\ToCards_03_21_25.txt" # logic to find today's notes goes here
todays_str = None
# with open(todays_notes_path, 'r', encoding='utf-8') as f:
#     todays_str = f.read()

cmd_dict = {"/t1": """What are PEP namespace packages?
Namespace packages allow multiple distributions to provide packages that exist in the same top-level namespace. For example, both packages A and B could provide a module namespace.foo and namespace.bar respectively, even if they are separate distributions.
"""}
src_test_inps = (
)
acr = AnkiCsvReader("C:/Users/Richie/AppData/Roaming/Anki2/User 1/collection.anki2", "ToCards_03_21_25_TopicCloze.csv")
acr.fast_forward_to_data()
omitted_fields = ("deck",)
for i, r in enumerate(acr, start=1):
    d = r.to_dict()
    for o in omitted_fields:
        if o in d: del d[o]
    cmd_dict[f"/a{i}"] = json.dumps(d, indent=2, ensure_ascii=False)

def prompt_model(inp):
    messages = []
    if SYS_OBJ != None:
        messages.append(SYS_OBJ)

    messages.append({"role": "user", "content": inp})
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    not_sure_what_do_this = model.generate(input_ids = inputs, pad_token_id=tokenizer.eos_token_id, streamer = text_collector, use_cache = True)
    out = text_collector.text
    text_collector.text = ""
    return out
    
if __name__ == "__main__":
    while True:
        text_collector.text = ""
        print("\nwhat would you like to say?\n")
        inp = ""
        while True:
            line = input("")
            if line in cmd_dict:
                inp = cmd_dict[line]
                break
            if line[-2:] == "//":
                inp += line[:-2]
                break
            inp += line + '\n'
        
        res = prompt_model(inp)
        print("===========")
        
        m = re.search(pat, res)
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
