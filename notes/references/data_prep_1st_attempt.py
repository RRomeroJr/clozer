import csv
import os
import pprint
import json
from typing import Dict, List
import pandas
from finetune_sys_prompt import finetune_sys_prompt
topic_cloze_fields = ("NoteType", "Text", "Topic", "hint_str", "hint_str_hides", "input")
exclose_fields = ("NoteType", "Text", "input")
field_map = {
    "TopicCloze":{e: i for i, e in enumerate(topic_cloze_fields)},
    "ExCloze":{e: i for i, e in enumerate(exclose_fields)}
}

io_prompts: Dict[str, List[List[Dict[str, str]]]] = {'conversations': []}
include_sys_prompt = True
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')
def main():
    path = 'traning_set.txt'
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        # rows_list = list(reader)
        # for row in reader:
        #     print(row)
        #     input('continue..')
        #     scroll()
        for row in reader:
            print(row)
            if not row[0] in field_map:
                print(row[0])
                continue
            inp_index = field_map[row[0]]['input']
            row_list = [{'from':'human', 'value': row[inp_index]}]
            # row_dict = {'input': row[inp_index], 'output':{}}
            row_dict = {}
            for v, i in field_map[row[0]].items():
                print(v)
                if v == 'input':
                    continue
                row_dict[v] = row[i]
            assistant_obj = {'from':'gpt', 'value': json.dumps(row_dict, indent=2, ensure_ascii=False)}
            row_list.append(assistant_obj)
            io_prompts['conversations'].append(row_list)
        with open('clozer_sharegpt.jsonl', 'w', encoding='UTF-8') as out:
            for d in io_prompts['conversations']:
                out.write(json.dumps(d, ensure_ascii=False) + '\n')
        with open('clozer_sharegpt_visualize.json', 'w', encoding='UTF-8') as out:
            out.write(json.dumps(io_prompts, indent=2, ensure_ascii=False))
        df = pandas.DataFrame(io_prompts)
        df.to_parquet('cloze.parquet')

        chatml_to_share = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}
        share_to_chatml = {v: k for k, v in chatml_to_share.items()}
        re_map: Dict[str, List[List[Dict[str, str]]]] = {"conversations": []}
        for convo in io_prompts['conversations']:
            re_map['conversations'].append([])
            current_convo = re_map['conversations'][-1]
            if include_sys_prompt:
                current_convo.append({"role":"system", "content": finetune_sys_prompt})
            for d in convo:
                current_convo.append({share_to_chatml.get(k, k): share_to_chatml.get(v, v) for k, v in d.items()})
        with open('clozer_chatml.json', 'w', encoding='UTF-8') as out:
            out.write(json.dumps(re_map, indent=2, ensure_ascii=False))
        df = pandas.DataFrame(re_map)
        df.to_parquet('cloze_chatml.parquet')

            # data = {'conversations': []}
            # for p in io_prompts:
            #     p["human"] = p['input']
            #     p['gpt'] = p['output']
            #     line = ['human'],p['gpt']]
            #     out.write
            # out.write(json.dumps(io_prompts, ensure_ascii=False))
if __name__ == "__main__":
    main()