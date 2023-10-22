import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, escape_quotes, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated json data (results of CKA step).", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")
    parser.add_argument("--vote-threshold", type=float, default=0, help="Threshold for votes.")

    args = parser.parse_args()
    data = read_json(args.datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        mturk_sample = {
            "id": sample["data_id"],
            "plausible_sentence": sample["plausible_sentence"],
            "implausible_sentence": sample["implausible_sentence"],
            "ck_options": []
        }
        
        kg_list = []

        for _, kg in enumerate(sample["knowledge"]+sample.get("new_knowledge", [])):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])
            votes = kg.get("votes")

            if kg_tuple not in kg_list and (votes is None or votes >= args.vote_threshold):
                mturk_sample["ck_options"].append({
                    "id": kg["id"], 
                    "head": escape_quotes(kg["head"]), 
                    "relation": kg["relation"], 
                    "tail": escape_quotes(kg["tail"])
                })
                kg_list.append(kg_tuple)
        
        mturk_data.append(mturk_sample)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_mt_exv{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")


if __name__ == "__main__":
    main()