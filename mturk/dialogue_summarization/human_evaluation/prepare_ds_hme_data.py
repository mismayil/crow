import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, mturk_prepare_dialogue, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to json data.", required=True)
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

        if sample["dialogue"]:
            mturk_data.append({
                "id": sample["data_id"],
                "dialogue": mturk_prepare_dialogue(sample["dialogue"]),
                "summary": sample["summary"],
                "summary_id": sample["hit_id"],
                "label": 1
            })

            for target in sample["alt_targets"]+sample["new_alt_targets"]:
                votes = target.get("votes")
                if votes is None or votes >= args.vote_threshold:
                    mturk_data.append({
                        "id": sample["data_id"],
                        "dialogue": mturk_prepare_dialogue(sample["dialogue"]),
                        "summary": target["text"],
                        "summary_id": target["id"],
                        "label": 0,
                    })

    output_path_stem = f"{args.output_dir}/mturk_data_ds_hme{args.suffix}"
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()