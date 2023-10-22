import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, parse_turn, mturk_prepare_dialogue, mturk_prepare_dialogue_turn, mturk_not_empty, chunk, escape_quotes, write_json, dim_from_relation

def find_turn_idx(dialogue, phrase):
    for index, turn in enumerate(dialogue):
        if phrase in turn:
            return index
    
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated data in json (results of CKV step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")
    parser.add_argument("--vote-threshold", type=int, default=2, help="Minimum number of votes for a KG to be included.")
    parser.add_argument("--batch-size", type=int, default=0, help="Number of samples per batch.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data before batching.")
    parser.add_argument("--turn-threshold", type=int, default=2, help="Minimum number of turns in a dialogue.")
    parser.add_argument("--cider", action="store_true", help="Use CIDER data directly.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = {}

    if args.cider:
        cider_data = []

        for sample in tqdm(data, desc="Preprocessing CIDER data"):
            cider_sample = {
                "data_id": f'cider-{sample["id"]}',
                "dialogue": sample["dialogue"],
                "knowledge": []
            }
            for index, kg in enumerate(sample["triplets"]):
                dim = dim_from_relation(kg["relation"], ignore_other=True)
                if dim:
                    cider_sample["knowledge"].append({
                        "id": f'{cider_sample["data_id"]}-{index}',
                        "head": kg["head"],
                        "relation": kg["relation"],
                        "tail": kg["tail"],
                        "head_turn": kg["head_index"]+1,
                        "tail_turn": kg["tail_index"]+1,
                    })
            
            if len(cider_sample["knowledge"]) > 0:
                cider_data.append(cider_sample)
        
        data = cider_data
    
    for sample in tqdm(data, desc="Preparing data"):
        for kg in sample["knowledge"]+sample.get("new_knowledge", []):
            votes = kg.get("votes", None)

            if votes is None or votes >= args.vote_threshold:
                head_turn_idx = kg.get("head_turn")
                tail_turn_idx = kg.get("tail_turn")

                if head_turn_idx is not None and mturk_not_empty(head_turn_idx):
                    head_turn_idx = int(head_turn_idx)-1
                else:
                    head_turn_idx = find_turn_idx(sample["dialogue"], kg["head"])

                if tail_turn_idx is not None and mturk_not_empty(tail_turn_idx):
                    tail_turn_idx = int(tail_turn_idx)-1
                else:
                    tail_turn_idx = find_turn_idx(sample["dialogue"], kg["tail"])

                if head_turn_idx > -1 and tail_turn_idx > -1 and (head_turn_idx != tail_turn_idx):
                    final_turn_idx = max(head_turn_idx, tail_turn_idx)

                    if final_turn_idx > args.turn_threshold:
                        mturk_sample = mturk_data.get((sample["data_id"], final_turn_idx))
                        
                        if mturk_sample is None:

                            mturk_sample = {
                                "id": sample["data_id"],
                                "dialogue": sample["dialogue"],
                                "knowledge": []
                            }
                            mturk_data[(sample["data_id"], final_turn_idx)] = mturk_sample

                        mturk_sample["knowledge"].append({
                            "id": kg["id"],
                            "head": kg["head"],
                            "relation": kg["relation"],
                            "tail": kg["tail"],
                            "head_turn": head_turn_idx+1,
                            "tail_turn": tail_turn_idx+1,
                        })
    
    for (data_id, final_turn_idx), mturk_sample in tqdm(mturk_data.items(), desc="Cleaning data"):
        dialogue = mturk_sample["dialogue"][:]
        kg_list = []
        filtered_ck_options = []

        for kg in mturk_sample["knowledge"]:
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])

            if kg_tuple not in kg_list:
                kg_list.append(kg_tuple)
                filtered_ck_options.append(kg)

                head_turn_idx = kg["head_turn"]-1
                tail_turn_idx = kg["tail_turn"]-1
                
                highlighted_head = f'<strong class="highlight">{kg["head"]}</strong>'
                highlighted_tail = f'<strong class="highlight">{kg["tail"]}</strong>'
                
                if highlighted_head not in dialogue[head_turn_idx]:
                    dialogue[head_turn_idx] = dialogue[head_turn_idx].replace(kg["head"], highlighted_head)
                
                if highlighted_tail not in dialogue[tail_turn_idx]:
                    dialogue[tail_turn_idx] = dialogue[tail_turn_idx].replace(kg["tail"], highlighted_tail)
                
                kg["head"] = escape_quotes(kg["head"])
                kg["tail"] = escape_quotes(kg["tail"])
        
        context_dialogue = dialogue[:final_turn_idx]

        final_turn = dialogue[final_turn_idx]
        final_turn_speaker, final_turn_utterance = parse_turn(final_turn)

        mturk_sample.update({
            "dialogue": mturk_prepare_dialogue(context_dialogue),
            "final_turn": mturk_prepare_dialogue_turn(final_turn_speaker, final_turn_utterance),
            "final_turn_prefix": f"{final_turn_speaker}:",
            "knowledge": filtered_ck_options
        })

    mturk_data = list(mturk_data.values())

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    if args.batch_size > 0:
        if args.shuffle:
            random.shuffle(mturk_data)
        mturk_data_batches = chunk(mturk_data, args.batch_size)
        for index, batch in enumerate(mturk_data_batches):
            pd.DataFrame(batch).to_csv(f"{args.output_dir}/mturk_data_odd_wsg{args.suffix}_batch{index+1}.csv", index=False)
    else:
        pd.DataFrame(mturk_data).to_csv(f"{args.output_dir}/mturk_data_odd_wsg{args.suffix}.csv", index=False)
    
    write_json(mturk_data, f"{args.output_dir}/mturk_data_odd_wsg{args.suffix}.json")

if __name__ == "__main__":
    main()