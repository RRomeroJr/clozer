import csv
import _csv
from dataclasses import dataclass
import math
import os
import pprint
from pprint import pformat, pprint
import json
import random
import re
import sys
from typing import Any, Dict, Iterable, Iterator, List
from typing import IO
import warnings

import anki.collection
from finetune_sys_prompt import finetune_sys_prompt
import pandas
from anki_helper_classes import _ClozeType, _NoteType, AnkiRow, TopicCloze, ExCloze
from data_helper_classes import ClassToDictEncoder, MsgExchange, MsgObj, Conversation, RRJRDataset, RRJRDatasetDict
from data_prep_funcs import *
from anki.collection import Collection as AnkiCollection
# import pandas

from dataclasses import dataclass, field
import csv
from typing import IO, Any
class RowLengthError(Exception):
    """Raised when a csv row has an unexpected number of columns"""
class UnexpectedValueError(Exception):
    """Raised when a csv found a value in a column number > the number of fields that the note has and isn't a tag"""

@dataclass
class AnkiCsvReader:
    anki_collection_path: str
    dataset_path: str = field(default=None)
    anki_collection: AnkiCollection = field(init=False)
    file: IO[Any] = field(init=False)
    reader: Iterator[List[str]] = field(init=False)
    guid_col: int = field(default=None, init=False)
    notetype_col: int = field(default=None, init=False)
    deck_col: int = field(default=None, init=False)
    max_cols: int = field(default=None, init=False)
    num_regex: re.Pattern = field(default=re.compile(r"[0-9][0-9]*"), init=False)
    col_arg_regex: re.Pattern = field(default=re.compile(r'#(\w+) column:(\d+)'), init=False)
    reader_offset: int = field(default=None, init=False)
    id_cols_dict: dict[str, int] = field(default=None, init=False)
    fields_dict: dict[str, dict[str, int]] = field(default=None, init=False)
    tags_start: int = field(default=None, init=False)
    data_start: int = field(default=None, init=False)
    test_guids: set = field(default=None, init=False)
    def __post_init__(self):
        self.anki_collection = AnkiCollection(self.anki_collection_path)
        if self.dataset_path != None:
            self.load()
    def load(self, dataset_path = None):
        if dataset_path != None :
            self.dataset_path = dataset_path
        self.file = open(self.dataset_path, 'r', encoding='UTF-8')
        
        print(f"opening {self.dataset_path}")
        self.reader = csv.reader(self.file, delimiter='\t')
        for row in self.reader:
            if row[0] == "":
                continue
            if row[0].startswith('#'):
                search = re.search(self.col_arg_regex, row[0])
                if not search:
                    print("skiping # row\n", row)
                    continue
                elif search[1] == "notetype":
                    print(search[0])
                    self.notetype_col = int(search[2]) - 1
                    print(self.notetype_col)
                elif search[1] == "deck":
                    print(search[0])
                    self.deck_col = int(search[2]) - 1
                elif search[1] == "guid":
                    print(search[0])
                    self.guid_col = int(search[2]) - 1
                elif search[1] == "tags":
                    print(search[0])
                    self.tags_start = int(search[2]) - 1
                elif search[1] == "test_guids":
                    print(search[0])
                    self.test_guids = set(search[2].strip().split(' '))
                    print('self.test_guids set to\n', self.test_guids)
                continue
            else:
                self.reader_offset = self.reader.line_num
                _id_cols = {"guid": self.guid_col, "notetype": self.notetype_col, "deck": self.deck_col, 'tags_start': self.tags_start}
                self.id_cols_dict = {k: v for k, v in _id_cols.items() if v != None}
                self.data_start = len(self.id_cols_dict)
                break
        # Check if we found everything we need
        if self.notetype_col == None:
            raise Exception(f"{self.__class__.__name__} couldn't find the notetype col. This is required for the reader to work properly")
        
        # restart the iterator. Fastforward to where we left off
        self.restart_iter()
        self.fast_forward_to_data()
        print("Checking data rows")
        self.fields_dict = {}
        for row in self.reader:
            self.check_data_row(row)
        print(f"Restarting iter for use")
        self.restart_iter()
        print(f"{self.__class__.__name__} load complete, {self.id_cols_dict}\n  {self.dataset_path},")
    def close(self):
        if hasattr(self, 'reader') and self.reader:
            del self.reader
        if hasattr(self, 'file') and self.file:
            self.file.close()
    def __del__(self):
        self.close()
    def g_col_decks(self):
        query = """Select name from decks"""
        return tuple(e[0].replace("\x1f", "::") for e in self.anki_collection.db.execute(query))
    def g_notetype_fields(self, nt_name) -> List[str]:
        query = f"""Select f.name From fields As f Inner Join notetypes nt On 
nt.id = f.ntid Where nt.name = '{nt_name}'"""
        fields: List[str] = [f[0] for f in self.anki_collection.db.execute(query)]
        return fields
    def restart_iter(self):
        self.file.close()
        self.file = open(self.dataset_path, 'r', encoding='UTF-8')
        self.reader = csv.reader(self.file, delimiter='\t')
    def fast_forward_to_data(self):
        print(f"reader fast forward till next = {self.reader_offset}")
        for _ in self.reader:
            # print(f"line num: {self.reader.line_num}")
            if self.reader.line_num >= self.reader_offset - 1:
                break
        print(f"fast forward complete, next will be {self.reader.line_num + 1}")
    def check_data_row(self, row):
        fields = self.g_notetype_fields(row[self.notetype_col])
        expected_len = len(fields) + len(self.id_cols_dict)
        
        if len(row) < expected_len:
            raise RowLengthError(f"line num {self.reader.line_num}, notetype {row[self.notetype_col]}, less than expected {expected_len} cols found {len(row)}")
        if self.tags_start != None:
            end = self.tags_start - 1
        else:
            end = len(row)
        
        for i in range(len(row), end):
            # all should be empty
            if row[i] != "":
                raise UnexpectedValueError(f"line num {self.reader.line_num}, notetype {row[self.notetype_col]},  found value at col {i + 1} but notetype should only have values in {len(expected_len)} cols\nlen fields{len(fields)} + len id_cols_dict {len(self.id_cols_dict)}")
        
        if row[self.notetype_col] not in self.fields_dict:
            new_dict = {k: v for k, v in self.id_cols_dict.items()}
            for i, f in enumerate(fields, start=len(self.id_cols_dict)):
                new_dict[f] = i
            self.fields_dict[row[self.notetype_col]] = new_dict
    def __iter__(self):
        self.i_count = -1
        return self
    def __next__(self):
        try:
            items = next(self.reader)
            self.i_count += 1
            return AnkiDataRow(items, self, self.i_count)
        except StopIteration:
            raise
@dataclass
class AnkiDataRow:
    items: list[str]
    acr: AnkiCsvReader
    iter_index: int = field(default=0, init=True)

    def __post_init__(self):
        try:
            self.notetype: str | None = self.items[self.acr.notetype_col]
        except (KeyError, IndexError, TypeError):
            self.notetype = None
        try:
            self.guid: str | None = self.items[self.acr.guid_col]
        except (KeyError, IndexError, TypeError):
            self.guid = None
        try:
            self.deck: str | None = self.items[self.acr.deck_col]
        except (KeyError, IndexError, TypeError):
            self.deck = None
    def get_note_id(self):
        "note id's are different from guids and the double as timestamps for the notes creation"
        if self.guid == None:
            raise RuntimeError(f"called get_note_id but {self.__class__.__name__} doesn't know it's guid. self.guid == {self.guid}")
        query = f"""select id from notes where guid = '{self.guid}'"""
        return self.acr.anki_collection.db.execute(query)[0][0]
    def tags(self) -> list[str] | None:
        try:
            return self.items[self.acr.tags_start:]
        except (KeyError, IndexError, TypeError):
            return None
    def data(self):
        return self.items[len(self.acr.id_cols_dict)-1:]
    def to_dict(self) -> dict[str, str]:
        return {field: self.items[index] for field, index in self.acr.fields_dict[self.items[self.acr.notetype_col]].items()}

    def __iter__(self):
        # Reset the index and return self as the iterator
        self.iter_index = 0
        return self
    def get_col(self, inp: str, default = None):
        if self.acr == None:
            warnings.warn(f"{self.__class__.__name__}.get_col called on a row with no associated AnkiCsvReader. returning default")
            return default
        if inp in self.acr.fields_dict[self.items[self.acr.notetype_col]]:
            return self.acr.fields_dict[self.items[self.acr.notetype_col]][inp]
        else:
            return default
    def get(self, inp: str, default = None):
        if self.acr == None:
            warnings.warn(f"{self.__class__.__name__}.get called on a row with no associated AnkiCsvReader. returning default")
            return default
        if inp in self.acr.fields_dict[self.items[self.acr.notetype_col]]:
            return self.items[self.acr.fields_dict[self.items[self.acr.notetype_col]][inp]]
        else:
            return default
    def __next__(self):
        if self.iter_index < len(self.items):
            result = self.items[self.iter_index]
            self.iter_index += 1
            return result
        else:
            raise StopIteration

