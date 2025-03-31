from datasets import Dataset, DatasetDict
import json
import os
import datasets
from datasets import load_dataset
from pprint import pprint, pformat
# Create datasets for each split
# combined = load_dataset('json', data_files={ 'train': 'convos_train.json', 'test': 'convos_test.json' }, field="conversations", split=None) 
combined = dataset = load_dataset('parquet', 
                      data_files={
                          'train': 'attempt2_train.parquet',
                          'test': 'attempt2_test.parquet'
                      },
                      split=None)
# conversations_dataset = combined.select_columns(["conversations"])
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')

print(combined)
print(combined['test'])
print(combined['train'])
print(combined.select_columns(["conversations"]))
print(pformat(combined['test'][0]['conversations']))
# print(pformat(combined["test"][0]["conversations"]))

