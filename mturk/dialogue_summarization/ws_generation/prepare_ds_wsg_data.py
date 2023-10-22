import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random
import string

sys.path.append("../../..")

from utils import read_json, mturk_prepare_dialogue, chunk, escape_quotes, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated data in json (results of CKV step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")
    parser.add_argument("--vote-threshold", type=int, default=2, help="Minimum number of votes for a KG to be included.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = []
    
    for sample in tqdm(data, desc="Preparing data"):
        kg_list = []
        mturk_sample = {
            "id": sample["data_id"],
            "dialogue": sample["dialogue"],
            "summary": sample["summary"],
            "knowledge": []
        }

        for kg in sample["knowledge"]+sample.get("new_knowledge", []):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])
            votes = kg.get("votes", None)

            if votes is None or votes >= args.vote_threshold:
                if kg_tuple not in kg_list:
                    kg_list.append(kg_tuple)

                    dialogue_phrase = kg["head"] if kg["head_from"] == "dialogue" else kg["tail"]
                    summary_phrase = kg["head"] if kg["head_from"] == "summary" else kg["tail"]
                    dialogue_phrase = dialogue_phrase.strip().strip(string.punctuation)
                    summary_phrase = summary_phrase.strip().strip(string.punctuation)
                    highlighted_dialogue_phrase = f'<strong class="highlight">{dialogue_phrase}</strong>'
                    highlighted_summary_phrase = f'<strong class="highlight">{summary_phrase}</strong>'

                    for turn_idx, turn in enumerate(mturk_sample["dialogue"]):
                        if dialogue_phrase in turn and highlighted_dialogue_phrase not in turn:
                            mturk_sample["dialogue"][turn_idx] = turn.replace(dialogue_phrase, highlighted_dialogue_phrase)
                    
                    if summary_phrase in mturk_sample["summary"] and highlighted_summary_phrase not in mturk_sample["summary"]:
                        mturk_sample["summary"] = mturk_sample["summary"].replace(summary_phrase, highlighted_summary_phrase)

                    mturk_sample["knowledge"].append({
                        "id": kg["id"],
                        "head": escape_quotes(kg["head"]),
                        "relation": kg["relation"],
                        "tail": escape_quotes(kg["tail"]),
                        "head_from": kg["head_from"],
                        "tail_from": kg["tail_from"],
                    })
        
        mturk_sample["dialogue"] = mturk_prepare_dialogue(mturk_sample["dialogue"])
        mturk_data.append(mturk_sample)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{args.output_dir}/mturk_data_ds_wsg{args.suffix}.csv", index=False)
    write_json(mturk_data, f"{args.output_dir}/mturk_data_ds_wsg{args.suffix}.json")

if __name__ == "__main__":
    main()