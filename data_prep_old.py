import csv
import _csv
import os
import pprint
import json
import random
import re
from typing import Any, Iterable
from finetune_sys_prompt import finetune_sys_prompt
import pandas

# import pandas
SYSTEM = 0
USER = 1
ASSISTANT = 2
NOTETYPE_COL = None
DECK_COL = None
max_cols = None
path = 'training_set2.txt'

topic_cloze_fields = ("Text", "Topic", "hint_index_str", "hint_index_str_shows", "input", "src_file")
topic_cloze_cols = ("id", "NoteType", "Deck") + topic_cloze_fields
excloze_fields = ("Text", "input", "src_file")
excloze_cols = ("id", "NoteType", "Deck") + excloze_fields
col_map = {
    "TopicCloze": topic_cloze_cols,
    "ExCloze": excloze_cols
}
field_map = {
    "TopicCloze": topic_cloze_fields,
    "ExCloze": excloze_fields
}
training_files  = {"HV_tips_and_tricks", "ToCards_03_02_25.odt", "ToCards_02_20_25.odt", "ToCards_01_23_25.odt", "ToCards_01_19_25.odt"}

# print("topic_cloze_cols", topic_cloze_cols)
io_prompts = {'conversations': []}
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')

def check_data_row(row: list[str], reader: _csv.reader):
# Make sure the notetype is expected
    assert row[NOTETYPE_COL] in col_map,\
    f"{row[NOTETYPE_COL]} not in col_map"

    # if there are tabs in one of the fields and as a result more cols then expected..
    assert len(row) <= max_cols, f"line {reader.line_num}: len {len(row)}, has more cols then the max_cols note {max_cols}\n{row}"
    
    
    r_col_count = real_col_count(row)
    assert r_col_count == len(col_map[row[NOTETYPE_COL]]),\
    f"line {reader.line_num}: cols {r_col_count} should be {len(col_map[row[NOTETYPE_COL]])}\n{pprint.pformat(row)}"

def main():
    global NOTETYPE_COL, DECK_COL, path, max_cols
    path = 'training_set2.txt'
    max_cols = len(max(col_map.values(), key=len, default=0))

    input_memo: dict[str, list[dict]]= {}
    unique_files:dict[str, list[dict]] = {}
    unique_files_train:dict[str, list[dict]] = {}
    unique_files_test:dict[str, list[dict]] = {}
    col_arg_regex = re.compile(r'#(\w+) column:(\d+)')
    print(f"max_cols is {max_cols}")
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        max_rows = -1
        count = 0 
        for row in reader:
            if max_rows >= 0 and count >= max_rows:
                break
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
                    DECK_COL = int(search[2]) - 1
                continue

            check_data_row(row, reader) # bunch of asserts. in here
            # all rows have correct number of cols and all notetypes are expected

            # mapping cols to col names
            r_dict = {k: row[i] for i, k in enumerate(col_map[row[NOTETYPE_COL]])}
            
            r_dict['Text'] = mk_seq_clozes(r_dict)
            unique_files.setdefault(r_dict['src_file'],[]).append(r_dict)
            if r_dict['src_file'] in training_files:
                unique_files_train.setdefault(r_dict['src_file'],[]).append(r_dict)
            else:
                unique_files_test.setdefault(r_dict['src_file'],[]).append(r_dict)

            count += 1
            print()
            print(count)

        print('all good')
        print(pprint.pformat(list(unique_files.keys())), f"\n{len(unique_files)}")
        scroll()
        # making the user, assistant data
        json_data:dict[str, dict[str,list[list[dict[str,str]]]]] = {}
        # making fake files by concating the inputs
        random.seed(99)
        for ufs in (unique_files_train, unique_files_test):
            if ufs == unique_files_train:
                split_name = "train"
            else:
                split_name = "test"
            json_data[split_name] = {"conversations": []}
            split = json_data[split_name]
            for fn, datas in ufs.items():
                print(f"Processing {fn} : {'train' if fn in unique_files_train else 'test'}")
                split["conversations"].append([])
                count = 1
                ffile_str = ""
                resp_objs = []
                random_spacing = False
                # I'm not consistant with \n's and -- so I randomized it 
                for d in datas:
                    if d['input'] not in input_memo and random_spacing and random.randint(1, 100) <= 35:
                        ffile_str += '\n'

                    ffile_str += d['input'] + '\n'
                    input_memo.setdefault(d["input"], []).append(d)

                    if d['input'] not in input_memo and random_spacing and random.randint(1, 100) <= 25:
                        ffile_str += "\n"
                    if count < len(datas):
                        if random.randint(1, 100) <= 50:
                            ffile_str += "--" + "\n"
                        else:
                            ffile_str += "-" * random.randint(2, 14) + "\n"
                    elif random.randint(1, 100) <= 30:
                        ffile_str += "--" + "\n" * random.randint(0,1)
                    resp_objs.append({k: v for k, v in d.items() if k != "input" and k != "src_file" and (k == 'NoteType' or k in field_map[d['NoteType']])})
                    count += 1
                # just to preview output
                split["conversations"][-1].append({"role":  "system", "content": finetune_sys_prompt})
                split["conversations"][-1].append({"role":  "user", "content": ffile_str})
                # pprint.pprint(resp_objs)
                # split["conversations"][-1].append({"role":  "assistant", "content": "\n".join(json.dumps(ro, indent=2, ensure_ascii=False) for ro in resp_objs)})
                split["conversations"][-1].append({"role":  "assistant", "content": resp_objs})
                with open(f'fake_files/{fn}', 'w+', encoding='UTF-8') as fake_file:
                    for e in split["conversations"][-1]:
                        curr_role = None
                        for k, v in e.items():
                            fake_file.write(f'{k}:')
                            if k == 'content':
                                if curr_role == 'user':
                                    fake_file.write('\n  ' + v.replace('\n', '\n  ') + '\n')
                                    # print(v)
                                else:
                                    fake_file.write(f" {len(v)}")
                                    fake_file.write('\n' + json.dumps(v, indent=2, ensure_ascii=False) + "\n\n")
                            else:
                                curr_role = v
                                fake_file.write(f"  {v}\n")

                with open(f'fake_files/{fn}.json', 'w+', encoding='UTF-8') as fake_file:
                    fake_file.write(json.dumps(split['conversations'][-1], indent=2, ensure_ascii=False))


            # Make the output json objects single line string
            for s, d in json_data.items():
                with open(f'convos_expanded_{s}.json', 'w+', encoding='UTF-8') as fake_file:
                    fake_file.write(json.dumps(d, indent=2, ensure_ascii=False))
                for convo in d["conversations"]:
                    convo[ASSISTANT]["content"] = json.dumps(convo[ASSISTANT]['content'], ensure_ascii=False)
                with open(f'convos_{s}.json', 'w+', encoding='UTF-8') as fake_file:
                    fake_file.write(json.dumps(d, indent=2, ensure_ascii=False))
        
                # fake_file.write(ffile_str)
                # fake_file.write('{\n  user: \"'+ ffile_str.replace("\n", "\n    ") + '\"\n}')
                # fake_file.write(json.dumps(json_data["conversations"][-1], indent=2, ensure_ascii=False))
                # read_test.valid_json_outputs_test()   
                df = pandas.DataFrame(d)
                df.to_parquet(f'attempt2_{s}.parquet')
                
def real_col_count(row: list):
    c = 0
    for r in row:
        if r == "" and c >= len(col_map[row[NOTETYPE_COL]]):
            continue
        c += 1
    return c
def mk_seq_clozes(r_dict: dict[str, str]):
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
    print('before\n', before, '\n\nafter\n', new_str)
    return new_str

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