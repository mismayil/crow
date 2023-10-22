import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, write_json

LANGUAGE_MAP = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "zh": "Chinese",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to json data.", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")
    parser.add_argument("--vote-threshold", type=float, default=3, help="Threshold for votes.")
    parser.add_argument("--size", type=int, default=100, help="Number of samples to select.")
    parser.add_argument("--source-lang", type=str, required=True)
    parser.add_argument("--target-lang", type=str, required=True)

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        if len(mturk_data) >= args.size:
            break

        for target in sample["targets"]:
            mturk_sample = {
                "id": sample["data_id"],
                "sentence": sample["source"],
                "translation": target["text"],
                "translation_id": sample["data_id"],
                "label": target["label"],
                "source_lang": LANGUAGE_MAP[args.source_lang],
                "target_lang": LANGUAGE_MAP[args.target_lang],
            }
            mturk_data.append(mturk_sample)

    output_path_stem = f"{args.output_dir}/mturk_data_mt_hme{args.suffix}"
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()