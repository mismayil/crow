import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import string

sys.path.append("../../..")

from utils import read_json, write_json, escape_quotes

def process_safetext(datapath):
    data = read_json(datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
            mturk_data.append({
                "id": sample["id"],
                "scenario": sample["scenario"].strip(string.punctuation),
                "safe_actions": [{**action, "text": escape_quotes(action["text"])} for action in sample["safe_actions"]],
                "unsafe_actions": [{**action, "text": escape_quotes(action["text"])} for action in sample["unsafe_actions"]]
            })
    
    return mturk_data

DATASET_PROCESSOR_MAP = {
    "safetext": process_safetext
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data.", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name.", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()
    mturk_data = DATASET_PROCESSOR_MAP[args.dataset](args.datapath)
    
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_sfd_cka{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()