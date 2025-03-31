import os
from typing import Iterable
vagueries = None
def load(additional_paths: Iterable = None):
    global vagueries
    if isinstance(additional_paths, str):
        additional_paths = (additional_paths,)
    paths = (
        f"{os.path.dirname(os.path.abspath(__file__))}/vagueries.txt",
        './vargueries/vagueries.txt',
        './vagueries.txt'
    )
    if isinstance(additional_paths, Iterable):
        paths = paths + tuple(e for e in additional_paths)
    for p in paths:
        if os.path.exists(p):
            print(f'loading vagueries from\n', p)
            with open(p, 'r', encoding='utf-8') as f:
                vagueries = tuple(line[:-1] for line in f)
    if not vagueries:
        print("couldn't vagueries tried:\n{}".format('  \n'.join(paths)))
load()
