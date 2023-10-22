import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, escape_quotes

def process_explagraph(datapath):
    data = read_json(datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        mturk_data.append({
            "id": sample["id"],
            "belief": sample["belief"],
            "argument": sample["argument"],
            "stance": sample["stance"],
            "knowledge": [{**kg, "head": escape_quotes(kg["head"]), "tail": escape_quotes(kg["tail"])} for kg in sample["knowledge"]]
        })
    
    return mturk_data

DATASET_PROCESSOR_MAP = {
    "explagraph": process_explagraph
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
    output_path_stem = f"{args.output_dir}/mturk_data_std_wsg{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()