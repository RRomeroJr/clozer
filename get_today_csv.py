import html
import csv, gc, csv, time, colorama, os, json, re

import torch
from anki_helper_classes import *
from finetune_sys_prompt import finetune_sys_prompt, deck_assign_sys_prompt
from typing import Dict, List, Type
from colorama import Fore, Back, Style
from anki_csv_reader import AnkiCsvReader
from anki_helper_classes import _NoteType
from us_helpers import delete_specific_directories, find_checkpoint
delete_specific_directories()
note_maker_dir = "note_maker_lora"
# note_maker_dir = "outputs/note_maker"
# deck_assigner_dir = "outputs/assigner" # what I've been calling it till 5/15/25
deck_assigner_dir = "deck_assigner_lora"
import sys
chpnt = find_checkpoint(
    chpnt_num=sys.argv[1] if len(sys.argv) > 2 else None, 
    chpnts_dir=note_maker_dir
)
delete_specific_directories()
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from transformers import PreTrainedModel
from peft.peft_model import PeftModelForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers import TextStreamer
from text_extraction.odt_text_extractor import *

pat = re.compile(r"<\|im_start\|>assistant\n(.*)<\|im_end\|>")
colorama.init()
model: PreTrainedModel = None
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name= chpnt["path"], 
    max_seq_length=8000,
    dtype = None,
    load_in_4bit = True
)

FastLanguageModel.for_inference(model)
print(f'{Fore.CYAN}After for_inference{Style.RESET_ALL}, tokenizer is now{Fore.GREEN}', type(tokenizer), f"{Style.RESET_ALL}")

tokenizer: LlamaTokenizerFast = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    map_eos_token = True
)
class TextCollector(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.text = ""
        
    def on_finalized_text(self, text, stream_end = False):
        self.text += text
        super().on_finalized_text(text, stream_end)
text_collector = TextCollector(tokenizer)
todays_notes_path = "D:\DocumentsHDD\ToCards\ToCards_05_02_25.odt" # logic to find today's notes goes here
today_file_name, today_file_ext = os.path.splitext(os.path.basename(todays_notes_path))

todays_str = extract_text_from_odt(todays_notes_path)
# with open(todays_notes_path, 'r', encoding='utf-8') as f:
#     todays_str = f.read()

def prompt_model(inp, system_prompt = None):
    messages = []
    if system_prompt != None:
        messages.append({"role": "system", "content": system_prompt})

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
    
    text_collector.text = ""
    prompt_out = prompt_model(todays_str, finetune_sys_prompt)
    print("===========")
    
    m = re.search(pat, prompt_out)
    if not m:
        raise Exception("When prompting for new notes, couldn't find assistant output")

    loaded: List[Dict] = None
    try:
        loaded = json.loads(m[1])
        # load_dump = json.dumps(loaded, indent=2, ensure_ascii=False)
        # print("Load, output:")
        # print(f"type: {type(loaded)},", load_dump)
        # print()
    except Exception as e:
        raise Exception (f"could not load output as json\n  {m[1]}") from e
    # we are not dead pog
    valid_noteTypes: tuple[Type[_NoteType]] = (ExCloze, TopicCloze)
    valid_names_to_types: dict[str, Type[_NoteType]] = {nt.__name__: nt for nt in valid_noteTypes}
    print("valid_names_to_types", valid_names_to_types)
    notetype_name_to_notes_dict: dict[str, list[dict[str, str]]] = {}
    note_list_dict: Dict[str, List[List[str]]] = {}
    total = 0
    for dict_obj in loaded: # loaded is an array of dicts
        for valid_nt in valid_noteTypes:
            if "notetype" in dict_obj and dict_obj["notetype"] == valid_nt.__name__:
                dict_obj["src_file"] = f"{today_file_name}{today_file_ext}"
                # deck = dict_obj.get("deck", deck_default)
                # note_fields_items = [valid_nt, deck]
                # note_fields_items = [valid_nt.__name__]
                
                # note_fields_items.extend(valid_nt.to_dict_to_list(dict_obj))
                print(f"{today_file_name}{today_file_ext}\ngot: {json.dumps(dict_obj, indent=2, ensure_ascii=False)}")
                notetype_name_to_notes_dict.setdefault(valid_nt.__name__, []).append(dict_obj)
                # print(f"adding following object gen'd from {today_file_name}{today_file_ext}\n{note_fiels_items}\nfrom: {json.dumps(dict_obj, indent=2, ensure_ascii=False)}")
                # note_list_dict.setdefault(valid_nt.__name__, []).append(note_fields_items)
                total += 1
    model.to("cpu")
    print("Dek-assigneri pradās")
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    chpnt = find_checkpoint(
        chpnt_num=None, 
        chpnts_dir=deck_assigner_dir
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name= chpnt["path"], 
        max_seq_length=8000,
        dtype = None,
        load_in_4bit = True
    )
    FastLanguageModel.for_inference(model)
    deck_default = "Active::AI_DEFAULT"
    temp = AnkiCsvReader("C:/Users/Richie/AppData/Roaming/Anki2/User 1/collection.anki2")
    decks_set = set(temp.g_col_decks())
    # print("decks_set", decks_set)
    temp.close()
    omitted_deck_keywords = {"Inactive", "Personal", "AlreadySeenBoost"}
    id_fields_order = ("guid", "notetype", "deck")
    del temp
    for nt_name, notes in notetype_name_to_notes_dict.items():
        new_dict_list: list[dict[str, str]] = []
        for n in notes:
            prompt_out = prompt_model(json.dumps(n, indent=2, ensure_ascii=False), deck_assign_sys_prompt)
            m = re.search(pat, prompt_out)
            if not m:
                raise Exception("When prompting for assigned deck, couldn't find assistant output")
            assigned_deck = m[1].strip()
            print("from ai:", repr(m[1]), "| striped: ", repr(assigned_deck))
            if assigned_deck in decks_set and not any(kw in assigned_deck for kw in omitted_deck_keywords):
                print("valid deck found assigning to ", assigned_deck)
                n["deck"] = assigned_deck
            else:
                if assigned_deck not in decks_set:
                    print(assigned_deck, "not in decks_set")
                else:
                    for kw in omitted_deck_keywords:
                        if kw in assigned_deck:
                            print(assigned_deck, "contains omitted deck keyword", kw)
                n["deck"] = deck_default
            # reorder keys so that they are easily writable
            new = {}
            for id_field in id_fields_order:
                if id_field in n:
                    new[id_field] = n[id_field]
            notetype = valid_names_to_types[n["notetype"]]
            for field_name in notetype.noteFields:
                if field_name in n:
                    new[field_name] = n[field_name]
                else:
                    new[field_name] = ""
            new_dict_list.append(new)
            print(new)
        notetype_name_to_notes_dict[nt_name] = new_dict_list

    new_notes_dir = "new_notes"
    new_csv_paths = []
    dont_encode = {"Text","guid","notetype","deck"}
    if total > 0:
        print(f"{total} arl{'ie' if total == 1 else 'ī'} not{'i' if total == 1 else 'ī'} syt ankot emi!")
        for type, note_list in notetype_name_to_notes_dict.items():
            csv_name = f"{today_file_name}_{type}.csv"
            csv_path = os.path.join(new_notes_dir, csv_name)
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_f:
                writer = csv.writer(csv_f, delimiter='\t')
                writer.writerow(["#separator:tab"])
                writer.writerow(["#html:true"])
                writer.writerow(["#notetype column:1"])
                writer.writerow(["#deck column:2"])
                for note in note_list:
                    writer.writerow( [val if field in dont_encode else html.escape(val) for field, val in note.items()] )
                new_csv_paths.append(csv_path)
        print(f"{csv_name} gōntaks!")
