import torch.version
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
model: str | nn.Module | PreTrainedModel = None
model, tokenizer = FastLanguageModel.from_pretrained(
model_name="mistralai/Mistral-7B-v0.1", 
max_seq_length=2048
)
print('After from pretrained, model is now', type(model))
# After from_pretrained, model is now <class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'> 
print(f'{Fore.CYAN}After from_pretrained{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
model = FastLanguageModel.get_peft_model(model)
print(f'{Fore.CYAN}After get_peft_model{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# After get_peft_model, model is now <class 'peft.peft_model.PeftModelForCausalLM'> 
tokenizer = get_chat_template(
tokenizer, 
mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
)
origdataset = load_dataset('parquet', data_files='D:\DocumentsHDD\Coding_Scrap\python\ollama\clozer\cloze.parquet', split='train') # use this
""" So I used parquet here to mimick the same structure as the example below from hf. But this function
can take in csv and json. If in the future I decide that making parquet is too annoy to do.

Or if this doesn't work."""
# origdataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")
conversations_dataset = origdataset.select_columns(['conversations'])

dataset = conversations_dataset.map(
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

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    dataset_num_proc = 1, # was told to change this by git hub for windows
    max_seq_length = 2048,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
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
print(f'{Fore.CYAN}After train(){Style.RESET_ALL}, model is now', type(model))
# After train(), model is now <class 'peft.peft_model.PeftModelForCausalLM'>
model: PeftModelForCausalLM = model
"""save_pretrained_gguf is being assigned to different things base off something.
It can be unsloth_save_pretrained_gguf or save_to_gguf_generic."""
model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method = "q4_k_m")