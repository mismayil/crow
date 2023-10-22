import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, escape_quotes, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated json data (results of WSG step).", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        if not sample["no_alt_target"]:
            mturk_sample = {
                "id": sample["data_id"],
                "belief": sample["belief"],
                "argument": sample["argument"],
                "stance": sample["stance"],
                "perturbation": sample["perturbation"],
                "knowledge": [{"id": kg["id"], "head": escape_quotes(kg["head"]), "relation": kg["relation"], "tail": escape_quotes(kg["tail"])} for kg in sample["knowledge"]],
                "ws_options": []
            }
            
            target_list = []

            for index, target in enumerate(sample["alt_targets"]):
                if target["text"] not in target_list:
                    target_list.append(target["text"])
                    mturk_sample["ws_options"].append({
                        **target,
                        "text": escape_quotes(target["text"])
                    })
            
            mturk_data.append(mturk_sample)

    output_path_stem = f"{args.output_dir}/mturk_data_std_wsv{args.suffix}"
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()