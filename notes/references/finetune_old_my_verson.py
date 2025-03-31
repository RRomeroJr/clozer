from datetime import datetime
import datasets
from us_helpers import delete_specific_directories, find_latest_checkpoint
delete_specific_directories()
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
from finetune_sys_prompt import finetune_sys_prompt
colorama.init()
MAX_SEQ_LENGTH = 8000
model: str | nn.Module | PreTrainedModel = None
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.1", 
    max_seq_length=MAX_SEQ_LENGTH
    # dtype = None,
    # load_in_4bit = True
)
print('After from pretrained, model is now', type(model))
# After from_pretrained, model is now <class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'> 
print(f'{Fore.CYAN}After from_pretrained{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
model = FastLanguageModel.get_peft_model(
    model
    # target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
print(f'{Fore.CYAN}After get_peft_model{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# After get_peft_model, model is now <class 'peft.peft_model.PeftModelForCausalLM'> 
tokenizer = get_chat_template(
    tokenizer,
    loftq_config = None
    # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
    # map_eos_token = True,
)
origdataset = dataset = load_dataset('parquet', 
                      data_files={
                          'train': 'attempt3_train.parquet',
                          'test': 'attempt3_test.parquet'
                      },
                      split=None) # use this was train before but I rll yshould be able to use None
# origdataset = load_dataset('parquet', data_files='D:\DocumentsHDD\Coding_Scrap\python\ollama\clozer\\attempt2.parquet',  split=None) # use this was train before but I rll yshould be able to use None
""" So I used parquet here to mimick the same structure as the example below from hf. But this function
can take in csv and json. If in the future I decide that making parquet is too annoying to do.

Or if this doesn't work."""

# origdataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
conversations_dataset = origdataset.select_columns(['conversations'])

dataset: datasets.DatasetDict = conversations_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=False,
            add_generation_prompt=False
        )
    },
    batched=True,
    batch_size=100,
    desc="Formatting conversations"
)
print("start" , datetime.now().strftime("%I:%M:%S %p"))
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset['train'],
    eval_dataset= dataset['test'],
    dataset_text_field = "text",
    dataset_num_proc = 1, # was told to change this by git hub for windows
    max_seq_length = MAX_SEQ_LENGTH,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        # eval_accumulation_steps = 4,
        # eval_strategy = "epoch",
        # eval_steps = 1,
        save_steps = 15,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 45,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),

)
trainer.train()
print("end trainning" , datetime.now().strftime("%I:%M:%S %p"))

print(f'{Fore.CYAN}After train(){Style.RESET_ALL}, model is now', type(model))
# After train(), model is now <class 'peft.peft_model.PeftModelForCausalLM'>
model: PeftModelForCausalLM = model
"""save_pretrained_gguf is being assigned to different things base off something.
It can be unsloth_save_pretrained_gguf or save_to_gguf_generic."""

model.generate
# model.save_pretrained("lora_model",)

# model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method = "q4_k_m")