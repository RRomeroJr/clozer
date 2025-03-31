import csv
import _csv
import math
import os
import pprint
from pprint import pformat, pprint
import json
import random
import re
import sys
from typing import Any, Dict, Iterable, List
from finetune_sys_prompt import finetune_sys_prompt
import pandas
from anki_helper_classes import _ClozeType, _NoteType, AnkiRow, TopicCloze, ExCloze
from data_helper_classes import ClassToDictEncoder, MsgExchange, MsgObj, Conversation, RRJRDataset, RRJRDatasetDict
from rrjr.rrjr_fm import g_seq_filename, sp_open
# import pandas


SYSTEM = 0
USER = 1
ASSISTANT = 2
NOTETYPE_COL = None
DECK_COL = None
GUID_COL = None
ID_COLUMNS = {}
max_cols = None
col_map: Dict[str, Dict[str, int]] = None
path = 'training_set2.txt'
notetypes = (TopicCloze, ExCloze)
field_map = {}
test_files  = {"ToCards_01_08_25.odt", "ToCards_1_10_25.odt", "ToCards_02_02_25.odt", "ToCards_02_22_25.odt", "ToCards_03_09_25.odt", "ToCards_01_18_25.odt"}
omited_fields = {"input", "src_file", "guid", "deck"}
dump_args = {"cls": ClassToDictEncoder, "indent": 2, "ensure_ascii": False}
num_regex = re.compile(r"[0-9][0-9]*")
# SYSTEM_MSG_OBJ = None
SYSTEM_MSG_OBJ = MsgObj("system", finetune_sys_prompt)
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')

def check_data_row(row: list[str], reader):
    # Make sure the notetype is expected
    assert row[NOTETYPE_COL] in col_map, f"{row[NOTETYPE_COL]} not in col_map"

    # if there are tabs in one of the fields and as a result more cols then expected..
    assert len(row) <= max_cols, f"line {reader.line_num}: len {len(row)}, has more cols then the max_cols note {max_cols}\n{row}"
    
    # """each line has the same number of rows as the notetype with the most amt of fields.
    # notetypes that have less than that have empty vals in those cols.
    # if we ignore those do we have the number of fields that we expect?"""
    r_col_count = real_col_count(row)
    assert r_col_count == len(col_map[row[NOTETYPE_COL]]), \
    f"line {reader.line_num}: cols {r_col_count} should be {len(col_map[row[NOTETYPE_COL]])}" + "\n[{}]".format('|\n'.join(row))
def mk_note_obj(row):
    if not NOTETYPE_COL:
        print("there must but a specified note col to make note objs")
        return
    notetype = next((nt for nt in notetypes if nt.__name__ == row[NOTETYPE_COL]), None)
    if not notetype:
        print(f"could not find matching notetype for {row[NOTETYPE_COL]}")

    args = {k: row[i] for k, i in ID_COLUMNS.items() if k != "notetype"}
    for i, k in enumerate(notetype.noteFields): args[k] = row[i + len(ID_COLUMNS)]

    return notetype(**args)

def main():
    global NOTETYPE_COL, DECK_COL, GUID_COL, ID_COLUMNS, path, max_cols, col_map
    path = 'training_set_3_5.txt'
    col_arg_regex = re.compile(r'#(\w+) column:(\d+)')
    startfrom = 0
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        print("reader line num", reader.line_num)
        for row in reader:
            if row[0] == "":
                continue
            if row[0].startswith('#'):
                search = re.search(col_arg_regex, row[0])
                if not search:
                    print("skiping # row\n", row)
                elif search[1] == "notetype":
                    print(search[0])
                    NOTETYPE_COL = int(search[2]) - 1
                elif search[1] == "deck":
                    print(search[0])
                    DECK_COL = int(search[2]) - 1
                elif search[1] == "guid":
                    print(search[0])
                    GUID_COL = int(search[2]) - 1
                continue
            elif not col_map:
                _id_cols = {"guid": GUID_COL, "notetype": NOTETYPE_COL, "deck": DECK_COL}
                print(_id_cols)
                ID_COLUMNS = {k: v for k, v in _id_cols.items() if v != None}
                print(ID_COLUMNS)
                offset = len(ID_COLUMNS)
                col_map = {}
                for nt in notetypes:
                    # adding the identifing cols 1st
                    col_map[nt.__name__] = {k: v for k, v in ID_COLUMNS.items() if v != None}
                    # then the note's fields
                    for i, nt_fieldname in enumerate(nt.noteFields):
                        col_map[nt.__name__][nt_fieldname] = offset + i
                startfrom = reader.line_num
                break
        
    print("col_map is ", pformat(col_map))
    print("reader line num should not be 0: ", reader.line_num)
    max_cols = len(max(col_map.values(), key=len, default=0))
    input_memo: dict[str, list[_NoteType]]= {}
    unique_files: dict[str, list[_NoteType]] = {}
    unique_files_train: dict[str, list[_NoteType]] = {}
    unique_files_test: dict[str, list[_NoteType]] = {}
    print(f"max_cols is {max_cols}")
    max_rows = -1
    count = 0
    include_system_prompt = False
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        print(reader.line_num)
        for _ in reader:
            print("speeding up", reader.line_num)
            # print("ffdsadf")
            if reader.line_num + 1 == startfrom:
                break
        for row in reader:
            # print("starting", reader.line_num)

            check_data_row(row, reader) # bunch of asserts. in here
            # all rows have correct number of cols and all notetypes are expected

            # mapping cols to col names
            nt_obj = mk_note_obj(row)
            # print("nt_obj type: ", type(nt_obj).__name__, row[:NOTETYPE_COL + 1])
            # r_dict = {k: row[i] for i, k in enumerate(col_map[row[NOTETYPE_COL]])}
            if isinstance(nt_obj, _ClozeType):
                mk_seq_clozes(nt_obj.__dict__)
                


            unique_files.setdefault(nt_obj.src_file, []).append(nt_obj)
            if nt_obj.src_file in test_files:
                unique_files_test.setdefault(nt_obj.src_file,[]).append(nt_obj)
            else:
                unique_files_train.setdefault(nt_obj.src_file,[]).append(nt_obj)

            count += 1

        print('all good')
        print(pformat(list(unique_files.keys())), f"\n{len(unique_files)}")
        scroll()
        # making the user, assistant data
        json_data:dict[str, dict[str,list[list[dict[str,str]]]]] = {}
        ds_obj = RRJRDatasetDict()
        ds_style2 = RRJRDatasetDict()
        # making fake files by concating the inputs
        random.seed(99)
        for ufs in (unique_files_train, unique_files_test):
            split_name = "train" if ufs == unique_files_train else "test"
            # json_data[split_name] = {"conversations": []}

            split = ds_obj.train if split_name == "train" else ds_obj.test
            for fn, note_objs in ufs.items():
                print(f"Processing {fn} : {'train' if fn in unique_files_train else 'test'}")
                # split["conversations"].append([])
                count = 1
                ffile_str = ""
                resp_objs = []
                random_spacing = False
                # I'm not consistant with \n's and -- so I randomized it 
                for note in note_objs:
                    if note.input not in input_memo:
                        if random_spacing and random.randint(1, 100) <= 35:
                            ffile_str += '\n'

                        ffile_str += note.input + '\n'
                        input_memo[note.input] = [note]

                        if random_spacing and random.randint(1, 100) <= 25:
                            ffile_str += "\n"
                        if count < len(note_objs):
                            if random.randint(1, 100) <= 50:
                                ffile_str += "--" + "\n"
                            else:
                                ffile_str += "-" * random.randint(2, 14) + "\n"
                        elif random.randint(1, 100) <= 30:
                            ffile_str += "--" + "\n" * random.randint(0,1)
                    else:
                        input_memo[note.input].append(note)
                    resp_objs.append({k: v for k, v in note.__dict__.items() if k not in omited_fields})
                    count += 1
                # just to preview output
                exchange = MsgExchange(MsgObj("user", ffile_str), MsgObj("assistant", resp_objs), SYSTEM_MSG_OBJ)
                convo = Conversation([exchange])
                split.conversations.append(convo)
                fpath = f'fake_files/{fn}'
                mk_simple_visualiztion_file(f'{fpath}.txt', split.conversations[-1])
                with open(f'{fpath}.json', 'w', encoding='UTF-8') as fake_file:
                    fake_file.write(json.dumps(split.conversations[-1], **dump_args))

                continue

        input_memo2: dict[str, list[_NoteType]]= {}
        for ufs in (unique_files_train, unique_files_test):
            split_name = "train" if ufs == unique_files_train else "test"
            split = ds_style2.g_split(split_name)
            for fn, note_objs in ufs.items():
                divisor = 2
                chunk_length =  math.ceil(len(note_objs) / divisor)
                note_split: List[List[_NoteType]] = []
                for i in range(0, len(note_objs), chunk_length):
                    note_split.append(note_objs[i: i + chunk_length])
                print([fn] + [len(ns) for ns in note_split], "was", len(note_objs))
                for count, ns in enumerate(note_split, start=1):
                    first = True
                    ffile_str = ""
                    for note in ns:
                        if note.input in input_memo2:
                            input_memo2[note.input].append(note)
                            continue
                        if not first:
                            if random.randint(1, 100) <= 50:
                                ffile_str += "\n--\n"
                            else:
                                ffile_str += "\n" + "-" * random.randint(2, 14) + "\n"
                        else:
                            first = False
                        ffile_str += note.input
                        input_memo2[note.input] = [note]
                    if random.randint(1, 100) <= 30:
                        ffile_str += "\n" + "--" + "\n" * random.randint(0,1)
                        # ffile_str is out "fake input"
                        # ns is our "fake output"
                        # filling up the new style dataset
                    resp_objs = [{field: val for field, val in note.__dict__.items() if field not in omited_fields} for note in ns]
                    exchange = MsgExchange(MsgObj("user", ffile_str), MsgObj("assistant", resp_objs), system = SYSTEM_MSG_OBJ)
                    convo = Conversation([exchange]) # still only 1 exchange per convo to save context
                    
                    split.conversations.append(convo)

                    # printing that specific file
                    fpath = "fake_files/" + fn + f"_div_{count}"
                    mk_simple_visualiztion_file(fpath + ".txt", split.conversations[-1])
                    with open(f'{fpath}.json', 'w', encoding='UTF-8') as fake_file:
                        fake_file.write(json.dumps(split.conversations[-1], **dump_args))
        print()
        scroll()
    
        # print with new style expanded for visual clarity
        for split_name, split in ds_style2.__dict__.items():
            with open(f'convos_newstyle_3_5_{split_name}_expanded.json', 'w', encoding='UTF-8') as fake_file:
                dump = json.dumps(split, **dump_args)
                fake_file.write(dump)

        # assistant objs becoming single strings
        for split_name, split in ds_style2.g_splits().items():
            for convo in split.conversations:
                for ex in convo.exchanges:
                    ex.assistant.content = json.dumps(ex.assistant.content, cls=ClassToDictEncoder, ensure_ascii=False)


        # print with new style
        for split_name, split in ds_style2.__dict__.items():
            with open(f'convos_newstyle_3_5_{split_name}.json', 'w', encoding='UTF-8') as fake_file:
                dump = json.dumps(split, **dump_args)
                fake_file.write(dump)
            re_loaded = json.loads(dump)
            df = pandas.DataFrame(re_loaded)
            df.to_parquet(f'attempt_3_5_{split_name}.parquet')
        return
        # old style generation didn't work but still here if needed. sonewhere ub there..
        # with open(f'convos_expanded_{s}.json', 'w+', encoding='UTF-8') as fake_file:
        #     fake_file.write(json.dumps(d, indent=2, ensure_ascii=False))
        # for convo in d["conversations"]:
        #     convo[ASSISTANT]["content"] = json.dumps(convo[ASSISTANT]['content'], ensure_ascii=False)
        # with open(f'convos_{s}.json', 'w+', encoding='UTF-8') as fake_file:
        #     fake_file.write(json.dumps(d, indent=2, ensure_ascii=False))
            # fake_file.write(ffile_str)
            # fake_file.write('{\n  user: \"'+ ffile_str.replace("\n", "\n    ") + '\"\n}')
            # fake_file.write(json.dumps(json_data["conversations"][-1], indent=2, ensure_ascii=False))
            # read_test.valid_json_outputs_test()

def mk_simple_visualiztion_file(fn: str, convo: Conversation):
    with open(fn, 'w', encoding='UTF-8') as fake_file:
        for exchange in convo.exchanges: # should only be 1
            ordered_roles = ("system", "user", "assistant")
            for role in ordered_roles:
                if exchange.g_role(role) == None:
                    continue
                fake_file.write(f'{role}:')
                if role == 'assistant':
                    # works because at this point it's just a list of dict
                    fake_file.write(f" {len(exchange.assistant.content)}")
                    fake_file.write('\n'
                        + json.dumps(exchange.assistant.content, **dump_args) + "\n\n")
                else:
                    fake_file.write('\n  ' + exchange.g_role(role).content.replace('\n', '\n  ') + '\n')
def mk_simple_visualiztion_json_files(fn, convo):
    pass
def real_col_count(row: list):
    c = 0
    for r in row:
        if r == "" and c >= len(col_map[row[NOTETYPE_COL]]):
            continue
        c += 1
    return c
def mk_seq_clozes(r_dict: dict[str, str]) -> dict[str, str]:
    before = r_dict['Text']
    cloze_pat = re.compile(r'(?:{{c(?P<cloze_num>\d+)::(?=.*?}})|!c(?P<hide_num>\d+))')
    matches = list(re.finditer(cloze_pat, r_dict["Text"]))
    
    memo = {}
    for m in matches:
        if not m['cloze_num']:
            continue
        memo.setdefault(m['cloze_num'], len(memo) + 1)
    view = [(m['cloze_num'], m[0]) for m in matches if m['cloze_num']]

    prev_end = 0
    new_str = ""
    for m in matches:
        new_str += r_dict['Text'][prev_end:m.start()]

        isHideTag = m['hide_num'] != None
        if isHideTag:
            cloze_num_key = 'hide_num'
            seq_num = memo.get(m[cloze_num_key], None)
        else:
            cloze_num_key = 'cloze_num'
            seq_num = memo[m[cloze_num_key]]
        
        if not seq_num and isHideTag:
            new_sub = ""
        else:
            new_sub = m[0][:m.start(cloze_num_key) - m.start()] + str(seq_num) + m[0][m.end(cloze_num_key) - m.start():]
        new_str += new_sub
        prev_end = m.end()

    # print(memo)
    new_str += r_dict['Text'][prev_end:]

    r_dict["Text"] = new_str
    # print('before\n', before, '\n\nafter\n', new_str)
    if "hint_index_str" in r_dict and r_dict["hint_index_str"] != "":
        org = r_dict["hint_index_str"]
        hint_index_arr = re.findall(num_regex, r_dict["hint_index_str"])
        for i in range(len(hint_index_arr)):
            if hint_index_arr[i] in memo:
                hint_index_arr[i] = memo[hint_index_arr[i]]
            r_dict["hint_index_str"] = " ".join(str(s) for s in hint_index_arr)

def clean():

    max_cols = len(max(col_map.values(), key=len, default=0))
    print(f"max_cols is {max_cols}")

    with open(PATH, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        count = 0
        for row in reader:
            if row[0] == "" or row[0].startswith('#'):
                continue
            nr = NoteRow(row, col_map[row[NOTETYPE_COL]], NOTETYPE_COL)
            if '\t' in nr["input"]:
                count += 1
            else:
                scroll()
                print(nr["input"])
                input()
            
        print(f"there are {count} notes that have tabs in the input")

if __name__ == "__main__":
    main()