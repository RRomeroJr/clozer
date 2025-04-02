import json
import re
from data_helper_classes import EncoderHelperBase
class AnkiRow(EncoderHelperBase):
    def __init__(self, notetype: str, guid: str = None, deck: str = None, input: str = None, scr_file: str = None):
        self.guid = guid
        self.notetype = notetype
        self.deck = deck
        self.input = input
        self.src_file = scr_file
class _NoteType(AnkiRow):
    noteFields = None
    def __init__(self, notetype, guid = None, deck = None, input = None, scr_file = None):
        super().__init__(notetype, guid, deck, input, scr_file)
    def __repr__(self):
        return self.__class__.__name__ + "()"
    @classmethod
    def to_dict_to_list(cls, inp_dict: dict) -> list[str]:
        res = [""] * len(cls.noteFields)
        print(inp_dict)
        for i, field in enumerate(cls.noteFields):
            print(f"checking {i}:", field)
            
            if field in inp_dict:
                print("inp_dict field found:", field)
                res[i] = inp_dict[field]
        return res
    def g_dict_in_row_order(self) -> dict[str]:
        id_fields = ("guid", "notetype", "deck")
        res = {}
        for field in id_fields:
            if field in self.__dict__:
                res[field] = self.__dict__[field]
        for_last = ("input", "src_file")
        for field in self.__class__.noteFields:
            if field not in res and field not in for_last:
                res[field] = self.__dict__[field]
        for field in for_last:
                res[field] = self.__dict__[field]
        return res
class _ClozeType(_NoteType):
    def __init__(self, notetype, Text, guid = None, deck = None, input = None, scr_file = None):
        super().__init__(notetype, guid, deck, input, scr_file)
        self.Text = Text
class TopicCloze(_ClozeType):
    noteFields = ("Text", "Topic", "hint_index_str", "hint_index_str_shows", "dont_shuffle", "input", "src_file")
    def __init__(self, Text: str, Topic: str, hint_index_str: str, hint_index_str_shows: str, dont_shuffle: str, guid: str =None, deck: str = None, input: str = None, src_file: str = None):
        super().__init__("TopicCloze", Text, guid, deck, input, src_file)
        self.Topic = Topic  # Remove the comma
        self.hint_index_str = hint_index_str  # Remove the comma
        self.hint_index_str_shows = hint_index_str_shows  # No comma here either
        self.dont_shuffle = dont_shuffle
    
class ExCloze(_ClozeType):
    noteFields = ("Text", "input", "src_file")
    def __init__(self, Text: str, guid=None, deck=None, input = None, src_file = None):
        super().__init__("ExCloze", Text, guid, deck, input, src_file)
