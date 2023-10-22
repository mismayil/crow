import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

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
            "scenario": sample["scenario"],
            "safe_actions": [{**action, "text": escape_quotes(action["text"])} for action in sample["safe_actions"]],
            "unsafe_actions": [{**action, "text": escape_quotes(action["text"])} for action in sample["unsafe_actions"]],
            "knowledge": []
        }

        for kg in sample["knowledge"]+sample.get("new_knowledge", []):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])
            votes = kg.get("votes", None)

            if votes is None or votes >= args.vote_threshold:
                if kg_tuple not in kg_list:
                    kg_list.append(kg_tuple)

                    mturk_sample["knowledge"].append({
                        "id": kg["id"],
                        "head": escape_quotes(kg["head"]),
                        "relation": kg["relation"],
                        "tail": escape_quotes(kg["tail"]),
                    })
        
        mturk_data.append(mturk_sample)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_sfd_wsg{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()