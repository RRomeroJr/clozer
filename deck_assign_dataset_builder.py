import csv
import json
import math
import os
import random
import re
import sys
import warnings

from settings.note_maker_settings import anki_col_path
from anki.collection import Collection
from pprint import pprint, pformat
from data_prep_funcs import mid_plus_minus_k
import pandas
from anki_csv_reader import AnkiCsvReader, AnkiDataRow
from data_helper_classes import *
from finetune_sys_prompt import deck_assign_sys_prompt
from rrjr_py import tri_split, ext_name
import statistics 
ntDict: dict[str, list] = {}
what_i_want = ("ExCloze", "TopicCloze")
omitted_fields = {"src", "src_file", "guid", "deck"}
omitted_decks = {"AlreadySeenBoost", "Personal", "AI_TEST"}
omited_notetypes = set()
dump_args = {"indent":2, "ensure_ascii": False}
def scroll():
    scroll_amt = os.get_terminal_size().lines -1
    print("\n" * scroll_amt + f"\033[{scroll_amt}A", end='')

def mk_datasets(
    tsv_overwrite: bool = False,
    use_v1: bool=False,
    regen_if_found: bool = True,
    dataset_path:str = "datasets/examples/deck-assigner/deck_assigner.tsv",
    middle_kx2 = 2
    ):
    acr = AnkiCsvReader(anki_col_path, dataset_path)
    print()
    # acr.fast_forward_to_data()
    deck_assign_dataset = RRJRDatasetDict()
    #region getting guids to make a test set
    deck_to_guids: dict[str, list[str]] = {}
    deck_to_convos: dict[str, list[Conversation]] = {}
    _test_guids_v1 = {r'DB{^T@51D_', r'eN[.P=q<Z*', r'uLwuIL/j8y', r'fRzPX|ilVi', r'c+hBg)v3_c', r'i%9N7G2)IL', r'tb?<I8%Sj,', r'OrefPin&,3', r'wKznz*4Vi]', r'qZK!8RkIKD', r'f%clsXcH36', r'uj#?YmCh@]', r'ueN1VnYvUT', r'ty,jC(da/+', r'QX`[`bRQXv', r'pqZ~UB?0a8', r'pr*K}.%ytL', r'fV`aYRL@N}', r'vBMJ`7:#[n', r'x3g(7T/}6!', r'kKx<.7|5Gd', r'b4aWp)k*#.', r'GN/CTSouP*', r'rnXf5T>O&e', r'g|iADb72:m', r'jHYP2,3zz3', r'rGhPX%!A<C', r'ETK<;$iT~R', r'vBQ^)Sq7~y', r'PDdHw%S0^:', r'uo!u#H3+w|', r'mUV}00?cmJ', r'p/,dg]n.xy', r'i$~wW$4bwA', r'IQQ>P<_ep5', r'M&f(Y%Iwv-', r'P7.`HB_!e9', r'tC)g(12;9-', r'E%d_Zbs0*L', r'dWRtu6ItJu', r'NWh2)Agpf8', r'x|K;&c]P~E', r'muQJV9F!=C', r'uBldLZM0m#', r'Q>e<qC~7rE', r'h+W!.Ca4$i', r'Hk1M#my]}F', r'A/QiazrTS?', r'A<1rJ9#ZHH', r'J]nY-V0W]Z', r'Ah|#gBy9-A', r'Ksi|5-[_?m', r'KsN0>n=5H;', r'Mi0T4}#uZ8', r'F#|%w90J!_', r'OpJ3xS%s+c', r'k%_Wq~W=Cm', r'b/)#aKyFte', r'v/i,@d^9(M', r'Or~PPZ85O:', r'k5/T:SWru9', r'KYD1&@6vjn', r'A#@-WIHg!d', r'I^8p,7CW0l', r'o0koe8%XQW', r'QznrNFk[$2', r'x?pZ<<dV5i', r't;*XA$F@tL', r'E:^45:#&F*', r'kX5VAovp<O', r'qAHUn4T?Ho', r'N~4zS0_c(^', r'Q[5&WyCu@G', r'v?vn1EtWv%', r'y@Gb059a+&', r'Iz%vn{s:{@', r'O$Z~BG6}^a', r'NqdgN}_tkt', r's~|nL&l8j', r'vZC!FX^18+', r'bb.&6oc8AZ', r'Hs7lG<erU_', r'E/(j?VB4OR', r'A9|g()wR~K', r'vsn]&qFmVI', r'HnC{09EF25', r'tWXewwLv5,', r'pBnO~eecN}', r'tHRcT!AQPK', r's*&W]i-/e(', r'}_>?aVUH!', r'Aim1jB9?l`', r'nY>J|^.mER', r'B?oy0[Ei7a', r'CsY-u+GuGX', r'rGuzVG8ATv', r'mLXLI8zb(m', r'Awa`=(XuN@', r'B8^Glj#[Gd', r'Od=[-jz<mz', r'us~.xJ~0DM', r'x;2|aArNF.', r'I,;&xrUtVm', r'eg4DeS8|Z3', r'uUASR;qdjc', r'kQQvQZ{ct-', r'l.u{[ebo>q', r'Elkr,}M[sd', r'e>SR;+A5y&'}
    old_set: set[str] = set(acr.headers['deck_assigner_test_guids'].split(' '))\
        if 'deck_assigner_test_guids' in acr.headers else None
    test_guids: set | None = old_set if old_set != None else None
    def _generate_test_guids():
        nonlocal test_guids
        if use_v1: # use the same guids from my first attempt
            test_guids = _test_guids_v1
        else: # get some guids to use as test set
            acr.restart_iter()
            acr.fast_forward_to_data()
            for count, row in enumerate(acr, start=1):
                if any(omitted_deck in row.deck for omitted_deck in omitted_decks):
                    print(f"deck with an illegal keyword found {row.deck}, skipping note")
                    continue
                deck_to_guids.setdefault(row.deck, []).append(row.guid)
            print()
            # for i, (deck, guids) in enumerate(sorted(deck_to_guids.items(), key=lambda e: len(e[1]), reverse=True ), start=1):
            #     print(f"{i}) {deck}: {len(guids) - min(len(guids)// 2, 2)}")
            # med = statistics.median([len(gl) - min(len(gl) //2 , 2) for gl in sorted(deck_to_guids.values(), key=lambda e: len(e), reverse=True )])
            # print(f"median {med}, len of keys {len(deck_to_guids)}, / 2 {len(deck_to_guids)/2}")
            # sys.exit()
            adjusted_dict: dict[str, list[str]] = {}
            for deck, guids in deck_to_guids.items():
                # If we have 1 we can't use it to test. we need it for training
                if len(guids) < 2: continue

                sample = random.sample(guids, min((len(guids)//6) + 1, 4))
                adjusted_dict[deck] = sample
            test_guids = set(guid for guid_list in adjusted_dict.values() for guid in guid_list)
                # We need it for the training set.
            # print("These are your test guids. Hardcode and look for them when building the dataset to add to test\n", "{" + ", ".join({f"r'{e}'" for e in deck_assigner_test_guids}) + "}")
            print("deck_to_guids------------")
            # for deck, guid_list in deck_to_guids.items():
                # print(f"{deck}: {len(guid_list)}")
            print(f"values: {sum(len(v) for v in deck_to_guids.values())}")
            print("adjusted_dict------------")
            # for deck, guid_list in adjusted_dict.items():
                # print(f"{deck}: {len(guid_list)}")
            print(f"values: {sum(len(v) for v in adjusted_dict.values())}")
            print("test_guids-----------------")
            print(f"values: {len(test_guids)}")
        acr.headers['deck_assigner_test_guids'] = " ".join(test_guids)
        if tsv_overwrite:
            rewrite_path = dataset_path
            acr.rewrite(overwrite=True)
        else:
            datasets_dir, filename, ext = tri_split(dataset_path)
            rewrite_path = os.path.join(datasets_dir, f"{filename}_rewrite{ext}").replace("\\", "/")
            acr.rewrite(out_path=rewrite_path)
            acr.load(rewrite_path)

    # test_guids: set | None = None
    if regen_if_found:
        print("Forcing regeneration even if found")
        _generate_test_guids()
    elif old_set == None:
        print("Generating guids because none found")
        _generate_test_guids()
    else:
        print("skipping guid generation")

    if test_guids == None: # if STILL None
        warnings.warn("No test_guids were found or generated. setting to an empty set")
        test_guids = set()
    if old_set != None:
        in_old_but_not_new = set(ele for ele in old_set if ele not in test_guids)
        print("new test_guids same as old. missing:" if len(in_old_but_not_new) <= 0 else
            "new test_guids different from old. missing:",
            f"{len(in_old_but_not_new)}/{len(old_set)}")
    else:
        print(f"No old set found. {len(test_guids)} new test guids")
    
    acr.restart_iter()
    acr.fast_forward_to_data()
    print("making actual datasets starting on line", acr.reader.line_num)
    for count, row in enumerate(acr, start=1):
        if any(omitted_deck in row.deck for omitted_deck in omitted_decks):
            print(f"convos builder found omitted deck {row.deck}, skipping note")
            continue
        # building a note without the fields I don't want it to make a decision on. Like deck, guid, src, src_file
        user_input_dict = {field: row.items[field_index] for field, field_index in acr.fields_dict[row.notetype].items() if field not in omitted_fields}
        # converting to a json object str. Not sure if I want the indent or not here.
        # for now it's here.
        # user_input = json.dumps(user_input_dict, ensure_ascii=False)
        user_input = json.dumps(user_input_dict, **dump_args)
        answer = row.deck
        # print(user_input, '\nanswer:', answer)
        convo = Conversation(MsgExchange(
            MsgObj("user", user_input),
            MsgObj("assistant", answer),
            MsgObj("system", deck_assign_sys_prompt)
        ))

        if row.guid in test_guids:
            deck_assign_dataset.test.conversations.append(convo)
        else:
            deck_to_convos.setdefault(row.deck, []).append(convo)
    # Was thinking of finding the median. There are 762 notes in Valy but 6 in regex.
    # This is too extreme. But with a amount of numbers the set. The median kind changes
    # too much when adding or removing decks. Here are a few options..

    mid_pm_k = mid_plus_minus_k([len(v) for v in deck_to_convos.values()], middle_kx2)
    # example_cap = 12 # What I think it was the 1st time
    example_cap = int(round(statistics.mean(mid_pm_k)))
    # example_cap = max(mid_pm_k) # currently 11
    print(f"new style median {example_cap}, mid +/- k {mid_pm_k}")
    for deck, convos in deck_to_convos.items():
        if len(convos) < example_cap:
            deck_assign_dataset.train.conversations.extend(convos)
        else:
            sample = random.sample(convos, max(example_cap, (len(convos) // 80)))
            deck_to_convos[deck] = sample
            deck_assign_dataset.train.conversations.extend(sample)
    # santity check print
    # print("deck and lengths -------")
    # for deck, convos in deck_to_convos.items():
    #     print(f"{deck}: {len(convos)}")
    # print("-----------------------")
    # writing json files and parquet
    datasets_dir, filename, _ = tri_split(dataset_path)
    for split, ds in deck_assign_dataset.__dict__.items():
        print(f"{filename}_{split}")
        dump = json.dumps(ds, cls=ClassToDictEncoder, **dump_args)
        # f_type = 'json'
        # json_path = os.path.join(datasets_dir, f"{filename}_{split}.{f_type}")
        # with open(json_path, 'w', encoding="UTF-8") as f:
        #     f.write(dump)
        f_type = "parquet"
        parquet_path = os.path.join(datasets_dir, f"{filename}_{split}.{f_type}")
        df = pandas.DataFrame(json.loads(dump))
        df.to_parquet(parquet_path)
def main():
    mk_datasets(tsv_overwrite=False, use_v1= False, regen_if_found=True)
if __name__ == "__main__":
    main()