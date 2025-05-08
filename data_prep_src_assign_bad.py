import csv
import json
import os
import random
import re
from anki.collection import Collection
from pprint import pprint, pformat

import pandas
from anki_csv_reader import AnkiCsvReader, AnkiDataRow
from data_helper_classes import *
from finetune_sys_prompt import deck_assign_sys_prompt
import statistics
import settings.note_maker_settings as settings

ntDict: dict[str, list] = {}
what_i_want = ("ExCloze", "TopicCloze")
omitted_fields = {"input", "src_file", "guid", "deck"}
omitted_decks = {"AlreadySeenBoost", "Personal", "AI_TEST"}
omited_notetypes = set()
dump_args = {"indent": 2, "ensure_ascii": False}
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')

def main():
    dataset_path = "datasets/anki_exports/training_set3_5.csv"
    acr = AnkiCsvReader(settings.anki_col_path, dataset_path)
    print()
    acr.fast_forward_to_data()
    deck_assign_dataset = RRJRDatasetDict()
    #region getting guids to make a test set
    test_files  = {"ToCards_01_08_25.odt", "ToCards_1_10_25.odt", "ToCards_02_02_25.odt", "ToCards_02_22_25.odt", "ToCards_03_09_25.odt", "ToCards_01_18_25.odt"}
    omitted_fields = {"guid", "src", "src_file"}
    deck_to_guids: dict[str, list[str]] = {}
    deck_to_convos: dict[str, list[Conversation]] = {}
    test_guids = {r'DB{^T@51D_', r'eN[.P=q<Z*', r'uLwuIL/j8y', r'fRzPX|ilVi', r'c+hBg)v3_c', r'i%9N7G2)IL', r'tb?<I8%Sj,', r'OrefPin&,3', r'wKznz*4Vi]', r'qZK!8RkIKD', r'f%clsXcH36', r'uj#?YmCh@]', r'ueN1VnYvUT', r'ty,jC(da/+', r'QX`[`bRQXv', r'pqZ~UB?0a8', r'pr*K}.%ytL', r'fV`aYRL@N}', r'vBMJ`7:#[n', r'x3g(7T/}6!', r'kKx<.7|5Gd', r'b4aWp)k*#.', r'GN/CTSouP*', r'rnXf5T>O&e', r'g|iADb72:m', r'jHYP2,3zz3', r'rGhPX%!A<C', r'ETK<;$iT~R', r'vBQ^)Sq7~y', r'PDdHw%S0^:', r'uo!u#H3+w|', r'mUV}00?cmJ', r'p/,dg]n.xy', r'i$~wW$4bwA', r'IQQ>P<_ep5', r'M&f(Y%Iwv-', r'P7.`HB_!e9', r'tC)g(12;9-', r'E%d_Zbs0*L', r'dWRtu6ItJu', r'NWh2)Agpf8', r'x|K;&c]P~E', r'muQJV9F!=C', r'uBldLZM0m#', r'Q>e<qC~7rE', r'h+W!.Ca4$i', r'Hk1M#my]}F', r'A/QiazrTS?', r'A<1rJ9#ZHH', r'J]nY-V0W]Z', r'Ah|#gBy9-A', r'Ksi|5-[_?m', r'KsN0>n=5H;', r'Mi0T4}#uZ8', r'F#|%w90J!_', r'OpJ3xS%s+c', r'k%_Wq~W=Cm', r'b/)#aKyFte', r'v/i,@d^9(M', r'Or~PPZ85O:', r'k5/T:SWru9', r'KYD1&@6vjn', r'A#@-WIHg!d', r'I^8p,7CW0l', r'o0koe8%XQW', r'QznrNFk[$2', r'x?pZ<<dV5i', r't;*XA$F@tL', r'E:^45:#&F*', r'kX5VAovp<O', r'qAHUn4T?Ho', r'N~4zS0_c(^', r'Q[5&WyCu@G', r'v?vn1EtWv%', r'y@Gb059a+&', r'Iz%vn{s:{@', r'O$Z~BG6}^a', r'NqdgN}_tkt', r's~|nL&l8j', r'vZC!FX^18+', r'bb.&6oc8AZ', r'Hs7lG<erU_', r'E/(j?VB4OR', r'A9|g()wR~K', r'vsn]&qFmVI', r'HnC{09EF25', r'tWXewwLv5,', r'pBnO~eecN}', r'tHRcT!AQPK', r's*&W]i-/e(', r'}_>?aVUH!', r'Aim1jB9?l`', 
r'nY>J|^.mER', r'B?oy0[Ei7a', r'CsY-u+GuGX', r'rGuzVG8ATv', r'mLXLI8zb(m', r'Awa`=(XuN@', r'B8^Glj#[Gd', r'Od=[-jz<mz', r'us~.xJ~0DM', r'x;2|aArNF.', r'I,;&xrUtVm', r'eg4DeS8|Z3', r'uUASR;qdjc', r'kQQvQZ{ct-', r'l.u{[ebo>q', r'Elkr,}M[sd', r'e>SR;+A5y&'}
    # test_guids = set()
    for row in acr:
        src = row.get("src")
        if src == None: continue
        inp_dict = {field: val for field, val in row.to_dict().items() if field not in omitted_fields}

        print("inp:\n{}".format(json.dumps(inp_dict, indent=2, ensure_ascii=False)))
        print(f"assistant resp:\n{src}")
        convo = Conversation(MsgExchange(
            MsgObj("user", inp_dict),
            MsgObj("assistant", src)
        ))
        print()
if __name__ == "__main__":
    main()