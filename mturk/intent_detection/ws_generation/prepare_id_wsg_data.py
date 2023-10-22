import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random
import string

sys.path.append("../../..")

from utils import read_json, escape_quotes, write_json

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
            "headline": sample["headline"],
            "intent": sample["intent"],
            "intent_id": sample["intent_id"],
            "knowledge": []
        }

        for kg in sample["knowledge"]+sample.get("new_knowledge", []):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])
            votes = kg.get("votes", None)

            if votes is None or votes >= args.vote_threshold:
                if kg_tuple not in kg_list:
                    kg_list.append(kg_tuple)

                    headline_phrase = kg["head"] if kg["head_from"] == "headline" else kg["tail"]
                    intent_phrase = kg["head"] if kg["head_from"] == "intent" else kg["tail"]
                    headline_phrase = headline_phrase.strip().strip(string.punctuation)
                    intent_phrase = intent_phrase.strip().strip(string.punctuation)
                    highlighted_headline_phrase = f'<strong class="highlight">{headline_phrase}</strong>'
                    highlighted_intent_phrase = f'<strong class="highlight">{intent_phrase}</strong>'

                    if headline_phrase in mturk_sample["headline"] and highlighted_headline_phrase not in sample["headline"]:
                        mturk_sample["headline"] = sample["headline"].replace(headline_phrase, highlighted_headline_phrase)
                    
                    if intent_phrase in mturk_sample["intent"] and highlighted_intent_phrase not in mturk_sample["intent"]:
                        mturk_sample["intent"] = mturk_sample["intent"].replace(intent_phrase, highlighted_intent_phrase)

                    mturk_sample["knowledge"].append({
                        "id": kg["id"],
                        "head": escape_quotes(kg["head"]),
                        "relation": kg["relation"],
                        "tail": escape_quotes(kg["tail"]),
                        "head_from": kg["head_from"],
                        "tail_from": kg["tail_from"],
                    })
        
        mturk_data.append(mturk_sample)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_id_wsg{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()