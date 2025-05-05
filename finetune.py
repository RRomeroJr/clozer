from dataclasses import dataclass, field
import gc
import json
import math
import os
from pprint import pformat
import subprocess
import threading
from typing import List, Optional
from us_helpers import delete_specific_directories, find_checkpoint
delete_specific_directories()

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import Trainer, TrainingArguments, TrainerCallback
import code
import traceback
from transformers import PreTrainedModel
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import sys
import colorama
from colorama import Fore, Back, Style
from peft.peft_model import PeftModelForCausalLM
from unsloth import UnslothTrainer, UnslothTrainingArguments
import data_prep
# import obj_to_src_data_prep2
import visualize
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
class MyCallbacks(TrainerCallback):
    def __init__(self):
        self.my_trainer: UnslothTrainer = None
        super().__init__()
colorama.init()
MAX_SEQ_LENGTH = 8192
# MAX_SEQ_LENGTH = 8000
model: str | nn.Module | PreTrainedModel = None
chpnt_dir = "outputs/note_maker"
if not os.path.exists(chpnt_dir):
    os.makedirs(chpnt_dir)
train_dataset = 'datasets/parquet/notes_to_json_train.parquet'
test_dataset = 'datasets/parquet/notes_to_json_test.parquet'
# rebuild_func = None
rebuild_func = data_prep.main
# process = subprocess.Popen(
#     [sys.executable, 'visualize.py', chpnt_dir],
#     start_new_session=True
# )
# monitor_thread = threading.Thread(
#     target=visualize.monitor_training,
#     args=(chpnt_dir,),
#     daemon=True  # This makes the thread exit when the main program exits
# )
# monitor_thread.start()
# rebuild_func = obj_to_src_data_prep2.main
ITERS = 18
EPOCHS = 1 # something even good to not make batch size weird
BATCH_SIZE = 1
GRADIENT_ACC_STEPS = 16
GPUS = 1 # maybe one day :(
TOTAL_BATCH_SIZE = GPUS * BATCH_SIZE * GRADIENT_ACC_STEPS
LEARNING_RATE = 1e-4
# LEARNING_RATE = 5e-5
print("LEARNING_RATE set to", LEARNING_RATE)
i = 0
while i < ITERS:
    print(f"{Fore.CYAN}this is iter{Style.RESET_ALL} {i + 1}")
    chpnt = find_checkpoint(chpnts_dir=chpnt_dir)
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
    
    if rebuild_func != None:
        rebuild_func()
    origdataset = load_dataset(
        'parquet',
        data_files={'train': train_dataset, 'test': test_dataset},
        split=None
    )
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
    DATASET_LEN = len(dataset["train"])
    print(f"train: math.ceil( ({len(dataset['train'])} / {TOTAL_BATCH_SIZE}) * {EPOCHS} )")
    print(f"test: math.ceil( ({len(dataset['test'])} / {TOTAL_BATCH_SIZE}) * {EPOCHS} )")
    EPOCH_IN_STEPS_TRAIN = int(math.ceil( (len(dataset["train"]) / TOTAL_BATCH_SIZE) ))
    print("EPOCH_IN_STEPS_TRAIN", EPOCH_IN_STEPS_TRAIN)
    EPOCH_IN_STEPS_TEST = int(math.ceil( (len(dataset["test"]) / TOTAL_BATCH_SIZE) ))
    print("EPOCH_IN_STEPS_TEST", EPOCH_IN_STEPS_TEST)
    print("chpnt num ", chpnt["num"] if chpnt else 0)
    EPOCHS_STEPS_TRAIN =(EPOCH_IN_STEPS_TRAIN * EPOCHS)
    EPOCHS_STEPS_TEST = EPOCH_IN_STEPS_TEST * EPOCHS
    EVAL_STEPS = 2
    MAX_STEP_ADD = ((DATASET_LEN // TOTAL_BATCH_SIZE) // 5) * 1
    MAX_STEPS = chpnt['trainer_state']['global_step'] + MAX_STEP_ADD if chpnt else MAX_STEP_ADD
    print("DATASET_LEN", DATASET_LEN, "MAX_STEP_ADD", MAX_STEP_ADD, "MAX_STEPS", MAX_STEPS)
    # MAX_STEPS = 60
    print("starting trainer.. seting max_steps to ", MAX_STEPS)
    # vvdoesn't seem to be working due to how unsloth is patching for backward compatibilityvv
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # # Calculate the total number of training steps
    # total_steps = MAX_STEPS
    # # Choose how many steps for the warmup phase (typically 10% of total steps)
    # warmup_steps = 0

    # # Create the polynomial decay scheduler
    # scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps,
    #     lr_end=1e-7,  # Final learning rate
    #     power=0.5,    # Polynomial power (2.0 = quadratic decay)
    # )
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # SFTTrainer
    my_callback = MyCallbacks()
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        dataset_text_field = "text",
        dataset_num_proc = 1, # was told to change this by git hub for windows
        max_seq_length = MAX_SEQ_LENGTH,
        packing = False, # Can make training 5x faster for short sequences.
        # optimizers=(optimizer, scheduler), # doesn't seem to be working due to how unsloth is patching for backward compatibility
        callbacks=[my_callback],
        # TrainingArguments
        args = TrainingArguments(
            # eval
            # eval goes thru the entire eval set no matter what
            do_eval=True, # unsure if I need this
            eval_accumulation_steps = GRADIENT_ACC_STEPS,
            eval_strategy = "steps",
            # eval_delay = 0,
            eval_steps = EVAL_STEPS, # only matters if set to "steps". logging steps + 1 good sometimes
            per_device_eval_batch_size=BATCH_SIZE,
            # train
            # num_train_epochs = chpnt["num"] + EPOCHS if chpnt else EPOCHS, #doesn't work. Name is in steps no matter what
            save_strategy = "steps",
            save_steps = EVAL_STEPS,
            max_steps = MAX_STEPS,
            per_device_train_batch_size = BATCH_SIZE,
            gradient_accumulation_steps = GRADIENT_ACC_STEPS,
            #learning rate stuff
            # embedding_learning_rate = 2e-5,
            # lr_scheduler_type = "constant",
            # lr_scheduler_type = "polynomial",
            # lr_scheduler_kwargs = {"power": 0.5}, # dones't seem to be working
            # lr_scheduler_type = "inverse_sqrt",
            lr_scheduler_type = "inverse_sqrt",
            learning_rate=LEARNING_RATE,          # Initial learning rate
            warmup_steps=1,
            # warmup_steps = EPOCH_IN_STEPS_TRAIN // 12,
            #logging
            logging_strategy="steps",
            logging_steps = 2, #only matters is logging_strategy="steps"
            #output
            output_dir = chpnt_dir,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.01,
            seed = 3407,
            report_to = "none", # Use this for WandB etc
        )
    )
    print("trainer type", type(trainer))
    my_callback.my_trainer = trainer
    print()
    print(f"{Fore.CYAN}trainer.args.lr_scheduler_type:{Style.RESET_ALL}", trainer.args.lr_scheduler_type, type(trainer.lr_scheduler))
    if chpnt == None and not os.path.exists(os.path.join(chpnt_dir, "initial_eval.json")):
        print(f"{Fore.CYAN}Can I get the learning rate this way?:{Style.RESET_ALL}", trainer.args.learning_rate)
        e_res = trainer.evaluate()
        e_res["initial_learning_rate"] = LEARNING_RATE
        
        with open(os.path.join(chpnt_dir, "initial_eval.json"), "w", encoding="UTF-8") as f:
            f.write(json.dumps(e_res, indent=2, ensure_ascii=False))
        print("Inital evaluate complete. Trying to train now causes an error to occur so exiting instead")
        sys.exit()
    # print(e_res)
    print(f"{Fore.CYAN}this is iter{Style.RESET_ALL} {i + 1}")
    # trainer.train(resume_from_checkpoint = False)
    trainer.train(resume_from_checkpoint = chpnt != None)
    print(f'{Fore.CYAN}After train(){Style.RESET_ALL}, model is now', type(model))
    del trainer
    # print("getting final eval..")
    # chpnt = find_checkpoint(silent = True, chpnts_dir=chpnt_dir)
    # try:
    #     with open(chpnt["path"] + "/trainer_state.json", "r", encoding="UTF-8") as f:
    #         last_trainer_state = json.loads(f.read())
    #     print(last_trainer_state["log_history"][-1])
    # except Exception as e:
    #     print(sys.exc_info)

    model.to("cpu")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    i += 1
# After train(), model is now <class 'peft.peft_model.PeftModelForCausalLM'>
"""save_pretrained_gguf is being assigned to different things base off something.
It can be unsloth_save_pretrained_gguf or save_to_gguf_generic."""
# model.save_pretrained("lora_model",)

# model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method = "q4_k_m")