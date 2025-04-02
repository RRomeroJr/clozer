import gc
import json
import math
from us_helpers import delete_specific_directories, find_checkpoint
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
from unsloth import UnslothTrainer, UnslothTrainingArguments
import data_prep
colorama.init()
MAX_SEQ_LENGTH = 8000
model: str | nn.Module | PreTrainedModel = None
chpnt = find_checkpoint()
base_model_path = "mistralai/Mistral-7B-v0.1"
if not chpnt:
    print(f"No checkpoint found loading {base_model_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= chpnt["path"] if chpnt else base_model_path,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True
)
# After from_pretrained, model is now <class 'transformers.models.mistral.modeling_mistral.MistralForCausalLM'> 
print(f'{Fore.CYAN}After from_pretrained{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
model = FastLanguageModel.get_peft_model(
    model,
    target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r= 16,
    lora_alpha = 16
)
print(f'{Fore.CYAN}After get_peft_model{Style.RESET_ALL}, model is now{Fore.GREEN}', type(model), f"{Style.RESET_ALL}")
# After get_peft_model, model is now <class 'peft.peft_model.PeftModelForCausalLM'> 
tokenizer = get_chat_template(
    tokenizer
    # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
)
ITERS = 1
EPOCHS = 1 # something even good to not make batch size weird
BATCH_SIZE = 2
GRADIENT_ACC_STEPS = 4
GPUS = 1 # maybe one day :(
TOTAL_BATCH_SIZE = GPUS * BATCH_SIZE * GRADIENT_ACC_STEPS
print()
for i in range(ITERS):
    data_prep.main()
    origdataset = load_dataset('parquet', 
                        data_files={
                            'train': 'attempt_3_5_train.parquet',
                            'test': 'attempt_3_5_test.parquet'
                        },
                        split=None)
    # origdataset = load_dataset('parquet', data_files='D:\DocumentsHDD\Coding_Scrap\python\ollama\clozer\cloze.parquet', split='train') # use this
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
    del conversations_dataset, origdataset
    gc.collect()
    print(f"train: math.ceil( ({len(dataset['train'])} / {TOTAL_BATCH_SIZE}) * {EPOCHS} )")
    print(f"test: math.ceil( ({len(dataset['test'])} / {TOTAL_BATCH_SIZE}) * {EPOCHS} )")
    EPOCH_IN_STEPS_TRAIN = int(math.ceil( (len(dataset["train"]) / TOTAL_BATCH_SIZE) ))
    print("EPOCH_IN_STEPS_TRAIN", EPOCH_IN_STEPS_TRAIN)
    EPOCH_IN_STEPS_TEST = int(math.ceil( (len(dataset["test"]) / TOTAL_BATCH_SIZE) ))
    print("EPOCH_IN_STEPS_TEST", EPOCH_IN_STEPS_TEST)
    print("chpnt num ", chpnt["num"] if chpnt else 0)
    EPOCHS_STEPS_TRAIN = EPOCH_IN_STEPS_TRAIN * EPOCHS
    EPOCHS_STEPS_TEST = EPOCH_IN_STEPS_TEST * EPOCHS
    print("starting trainer.. seting max_steps to ", chpnt["num"] + (EPOCHS_STEPS_TRAIN) if chpnt else (EPOCHS_STEPS_TRAIN))
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        dataset_num_proc = 1, # was told to change this by git hub for windows
        max_seq_length = MAX_SEQ_LENGTH,
        packing = False, # Can make training 5x faster for short sequences.
        # TrainingArguments
        args = UnslothTrainingArguments(
            # eval
            # eval goes thru the entire eval set no matter what
            do_eval=True, # unsure if I need this
            eval_accumulation_steps = GRADIENT_ACC_STEPS,
            eval_strategy = "steps",
            # eval_delay = 0,
            eval_steps = EPOCHS_STEPS_TRAIN, # only matters if set to "steps"
            per_device_eval_batch_size=BATCH_SIZE,
            # train
            warmup_steps = 5,
            # num_train_epochs = chpnt["num"] + EPOCHS if chpnt else EPOCHS, #doesn't work. Name is in steps no matter what
            save_strategy = "steps",
            save_steps = EPOCHS_STEPS_TRAIN,
            max_steps = chpnt["num"] + EPOCHS_STEPS_TRAIN if chpnt else EPOCHS_STEPS_TRAIN,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACC_STEPS,
            embedding_learning_rate = 2e-5,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_strategy="steps",
            logging_steps = EPOCH_IN_STEPS_TRAIN, #only matters is logging_strategy="steps"
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        )
    )
    print()
    # e_res = trainer.evaluate()
    # print(e_res)
    trainer.train(resume_from_checkpoint = chpnt != None)
    del trainer
    print("getting final eval..")
    chpnt = find_checkpoint(silent = True)
    with open(chpnt["path"] + "/trainer_state.json", "r", encoding="UTF-8") as f:
        last_trainer_state = json.loads(f.read())
    print(last_trainer_state["log_history"][-1])
print(f'{Fore.CYAN}After train(){Style.RESET_ALL}, model is now', type(model))
# After train(), model is now <class 'peft.peft_model.PeftModelForCausalLM'>
"""save_pretrained_gguf is being assigned to different things base off something.
It can be unsloth_save_pretrained_gguf or save_to_gguf_generic."""
# model.save_pretrained("lora_model",)

# model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method = "q4_k_m")