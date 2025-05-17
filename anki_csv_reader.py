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
from rrjr_py import ellipsis_truc, tri_split
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
    max_cols: int = field(default=None, init=False)
    num_regex: re.Pattern = field(default=re.compile(r"[0-9][0-9]*"), init=False)
    anki_key_regex: re.Pattern = field(default=re.compile(r'^#(.*?):(.*)$'), init=False)
    reader_offset: int = field(default=None, init=False)
    id_cols_dict: dict[str, int] = field(default=None, init=False)
    fields_dict: dict[str, dict[str, int]] = field(default=None, init=False)
    data_start: int = field(default=None, init=False)
    test_guids: set = field(default=None, init=False)
    headers: dict[str,str] = field(default_factory=dict, init=False)
    @property
    def guid_col(self):
        try:
            return self.headers["guid column"] - 1
        except KeyError: return None
    @guid_col.setter
    def guid_col(self, inp: int):
        if not isinstance(inp, int):
            raise ValueError("guid_col value must be an int")
        self.headers["guid column"] = inp + 1

    @property
    def notetype_col(self):
        try:
            return self.headers["notetype column"] - 1
        except KeyError: return None
    @notetype_col.setter
    def notetype_col(self, inp: int):
        if not isinstance(inp, int):
            raise ValueError("notetype_col value must be an int")
        self.headers["notetype column"] = inp + 1

    @property
    def deck_col(self):
        try:
            return self.headers["deck column"] - 1
        except KeyError: return None
    @deck_col.setter
    def deck_col(self, inp: int):
        if not isinstance(inp, int):
            raise ValueError("deck_col value must be an int")
        self.headers["deck column"] = inp + 1

    @property
    def tags_start(self):
        try:
            return self.headers["tags"] - 1
        except KeyError: return None
    @tags_start.setter
    def tags_start(self, inp: int):
        if not isinstance(inp, int):
            raise ValueError("tags_start value must be an int")
        self.headers["tags"] = inp + 1

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
                search = re.search(self.anki_key_regex, row[0])
                if not search:
                    print("skiping # row\n", row)
                    continue
                value_stripped = search[2].strip()
                val = value_stripped
                if search[1] == "guid column" or search[1] == "deck column" or search[1] == "notetype column" or search[1] == "tags":
                    val = int(value_stripped)

                print(f"#{search[1]}:{ellipsis_truc(val)}")
                self.headers[search[1]] = val
                continue
            else:
                self.reader_offset = self.reader.line_num
                _id_cols = {"guid": self.guid_col, "notetype": self.notetype_col, "deck": self.deck_col, 'tags_start': self.tags_start}
                print(_id_cols)
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
    def rewrite(self, out_path=None, overwrite=False):
        if out_path is None and not overwrite:
            ValueError("You must provide an out path or set overwrite to True")
        head, fn, ext = (None, None, None)
        rewrite_path = None
        if overwrite:
            head, fn, ext = tri_split(self.dataset_path)
            out_path = os.path.join(head, f"{fn}_rewrite_temp{ext}")
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            _writter = csv.writer(f, delimiter='\t')
            _writter.writerows([['#separator:tab'], ['#html:true']])
            for k, v in self.headers.items():
                if k == 'separator' or k == 'html': continue
                _writter.writerow([f'#{k}:{v}'])
            self.restart_iter()
            self.fast_forward_to_data()
            for row in self.reader:
                _writter.writerow(row)
        print("Closing old file\n", self.dataset_path)
        self.file.close()
        if overwrite:
            print("Overwriting old file")
            os.replace(out_path, self.dataset_path)
        print("Reloading overwritten file")
        self.load(self.dataset_path)
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

