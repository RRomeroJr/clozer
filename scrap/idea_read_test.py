import csv
import _csv
import os
import pprint
import json
import random
import re
from typing import Any, Iterable
USER = 0
ASSISTANT = 1

def valid_json_outputs_test(): # will not work in the "chunk style"
    class TestOutput():
        def __init__(self, output, role_obj, convo_num, robj_num, split = None):
            self.output = output
            self.role_obj = role_obj
            self.convo_num = convo_num
            self.robj_num = robj_num
            self.split = split
    data = None
    passed: list[TestOutput] = []
    tested: list[TestOutput] = []
    failed: list[TestOutput] = []
    with open('convos.json', 'r', encoding='UTF-8') as file:
        data:dict = json.loads(file.read())
    for split, d in data.items():
        for convo_num, convo_array in enumerate(d["conversations"]):
            for robj_num, role_obj in enumerate(convo_array):
                if role_obj['role'] != 'assistant': continue
                to = None
                try:
                    to = TestOutput(json.loads(role_obj['content']), role_obj, convo_num, robj_num, split=split)
                    passed.append(to)

                except Exception as e:
                    to = TestOutput(e, role_obj, convo_num, robj_num, split=split)
                    failed.append(to)

                tested.append(to)

    if len(failed) <= 0:
        print(f"all assistant outputs are readable by json loads")
        # print('\n-----\n'.join("{} split: {}\n\n{}".format(type(p.role_obj['content']), p.split, p.role_obj['content']) for p  in passed))
    else:
        print('\n-----\n'.join("{} split: {}\n\n{}".format(type(f.role_obj['content']), f.split, f.role_obj['content']) for f  in failed))

    print(f"FAILUES: {len(failed)}, Passed: {len(passed)}, Total {len(tested)}")

def unescape(s):
    """
    Unescapes characters in a string.
    
    Args:
        s (str): The string with escaped characters.
        
    Returns:
        str: The unescaped string.
    """
    # Dictionary mapping escape sequences to their actual representation
    escape_chars = {
        '\\\\': '\\',    # Backslash
        '\\n': '\n',     # Newline
        '\\r': '\r',     # Carriage return
        '\\t': '\t',     # Tab
        '\\b': '\b',     # Backspace
        '\\f': '\f',     # Form feed
        '\\"': '"',      # Double quote
        "\\'": "'",      # Single quote
        '\\a': '\a',     # Bell
        '\\v': '\v'      # Vertical tab
    }
    
    result = s
    for escaped, unescaped in escape_chars.items():
        result = result.replace(escaped, unescaped)
    
    # Handle octal escapes like \ooo
    import re
    result = re.sub(r'\\([0-7]{1,3})', 
                   lambda x: chr(int(x.group(1), 8)), 
                   result)
    
    # Handle hex escapes like \xhh
    result = re.sub(r'\\x([0-9a-fA-F]{2})', 
                   lambda x: chr(int(x.group(1), 16)), 
                   result)
    
    # Handle unicode escapes like \uxxxx
    result = re.sub(r'\\u([0-9a-fA-F]{4})', 
                   lambda x: chr(int(x.group(1), 16)), 
                   result)
    
    # Handle unicode escapes like \Uxxxxxxxx (for characters outside BMP)
    result = re.sub(r'\\U([0-9a-fA-F]{8})', 
                   lambda x: chr(int(x.group(1), 16)), 
                   result)
    
    return result
def main():
    valid_json_outputs_test()
if __name__ == "__main__":
    main()