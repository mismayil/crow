import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random
import re

sys.path.append("../../..")

from utils import read_json, write_json

def clean(text):
    return re.sub("^\d+\.", "", text.strip()).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to eval json data.", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")
    parser.add_argument("--vote-threshold", type=float, default=3, help="Threshold for votes.")
    parser.add_argument("--size", type=int, default=100, help="Number of samples to select.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        if len(mturk_data) >= args.size:
            break

        for instance in sample["instances"]:
            votes = instance.get("votes")

            if votes is None or votes >= args.vote_threshold:
                mturk_data.append({
                    "id": sample["data_id"],
                    "belief": instance["belief"],
                    "argument": instance["argument"],
                    "instance_id": instance["id"],
                    "label": 1 if instance["stance"] == "support" else 0
                })

    output_path_stem = f"{args.output_dir}/mturk_data_std_hme{args.suffix}"
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()