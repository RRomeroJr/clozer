import json
from pprint import pformat
import re
from typing import Dict, List
class EncoderHelperBase():
    pass
class ClassToDictEncoder(json.JSONEncoder):
    def default(self, obj):
        # Only convert to dict if it's a subclass of EncoderHelperBase

        if isinstance(obj, MsgExchange):
            return tuple(v for v in obj.__dict__.values() if v != None)
        if isinstance(obj, Conversation):
            # print(obj.exchanges)

            return tuple(y for exchange in obj.exchanges for y in self.default(exchange))
        elif isinstance(obj, EncoderHelperBase):
            return obj.__dict__
        return super().default(obj)
class MsgObj(EncoderHelperBase):
    def __init__(self, role: str, content: str | dict | List[dict]):
        self.role = role
        self.content = content
        pass
    def __repr__(self):
        indent = 4

        return self.__class__.__name__ + "(" \
        "\n"+ " " * indent + f"role: {self.role}" + \
        "\n" + " " * indent + f"content: " +\
        ("\n" + repr(self.content)).replace('\n', '\n' + ' ' * indent * 2) + \
        "\n)"

# )""".format(self.role, (("\n" + " " * indent * 2) if isinstance(self.content, list) or isinstance(self.content, dict) else "") + pformat(self.content).replace('\n', '\n' + ' ' * indent * 2))
class MsgExchange():
    def __init__(self, user: MsgObj, assistant: MsgObj, system: MsgObj = None):
        self.system = system
        self.user = user
        self.assistant = assistant
        pass
    def g_role(self, key) -> MsgObj:
        return self.__dict__.get(key, None)
    def __repr__(self):
        indent = 4
        r = self.__class__.__name__ + "(\n"
        if self.system:
            r += " " * indent + repr(self.system).replace("\n", "\n" + " " * indent) + "\n"
        if self.user:
            r += " " * indent + repr(self.user).replace("\n", "\n" + " " * indent) + "\n"
        if self.assistant:
            r += " " * indent + repr(self.assistant).replace("\n", "\n" + " " * indent) + "\n"
        return r + ")"

class Conversation():
    def __init__(self, exchanges: MsgExchange | list[MsgExchange] = None):
        
        if exchanges == None:
            self.exchanges = []
        elif isinstance(exchanges, List):
            self.exchanges = exchanges
        elif isinstance(exchanges, MsgExchange):
            self.exchanges = [exchanges]
        else:
            raise Exception("Conversation.exchanges must be MsgExchange | list[MsgExchange]")
    def __repr__(self):
        return self.__class__.__name__ + \
"""(
    {}
)""".format('\n'.join(repr(e).replace('\n', '\n    ') for e in self.exchanges))
class RRJRDataset(EncoderHelperBase):
    def __init__(self, conversations: list[Conversation] = None):
        if conversations == None:
            self.conversations = []
        else:
            self.conversations = conversations
    def __repr__(self):
        return self.__class__.__name__ + "()"
class RRJRDatasetDict(EncoderHelperBase):
    def __init__(self, train: RRJRDataset = None, test: RRJRDataset = None):
        if train == None: self.train = RRJRDataset()
        else: self.train = train
        if test == None: self.test = RRJRDataset()
        else: self.test = test
    def g_split(self, key) -> RRJRDataset:
        return self.__dict__.get(key, None)
    def g_splits(self) -> Dict[str, RRJRDataset]:
        return dict(self.__dict__.items())
    def __repr__(self):
        return self.__class__.__name__ + "()"