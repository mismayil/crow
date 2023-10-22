import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, clean, parse_turn, write_json, mturk_prepare_dialogue

def process_cider(datapath, num_samples=0):
    data = read_json(datapath)

    if num_samples > 0:
        data = random.sample(data, num_samples)
    
    processed_data = []

    for sample in tqdm(data, desc="Processing data"):    
        processed_data.append({
            "id": sample["id"],
            "dialogue": sample["dialogue"]
        })
    
    return processed_data

def process_wow(datapath, num_samples=0):
    data = read_json(datapath)

    if num_samples > 0:
        data = random.sample(data, num_samples)
    
    processed_data = []

    for sample in tqdm(data, desc="Processing data"):
        dialogue = []

        for turn in sample["dialogue"]:
            speaker = "#Person1#" if "Wizard" in turn["speaker"] else "#Person2#"
            dialogue.append({"speaker": speaker, "utterance": clean(turn["text"])})
    
        processed_data.append({
            "id": sample["id"],
            "dialogue": dialogue
        })
    
    return processed_data

def process_cmsd(datapath, num_samples=0):
    data = read_json(datapath)

    if num_samples > 0:
        data = random.sample(data, num_samples)
    
    processed_data = []

    for sample in tqdm(data, desc="Processing data"):
        dialogue = []

        for index, turn in enumerate(sample["turns"]):
            dialogue.append({"speaker": "#Person1#" if index % 2 == 0 else "#Person2#", "utterance": clean(turn)})
    
        processed_data.append({
            "id": sample["id"],
            "dialogue": dialogue
        })
    
    return processed_data

def process_reddit(datapath, num_samples=0):
    data = read_json(datapath)

    if num_samples > 0:
        data = random.sample(data, num_samples)
    
    processed_data = []

    for sample in tqdm(data, desc="Processing data"):
        dialogue = []

        for index, turn in enumerate(sample["dialogue"]):
            dialogue.append({"speaker": "#Person1#" if index % 2 == 0 else "#Person2#", "utterance": clean(turn)})
    
        processed_data.append({
            "id": sample["id"],
            "dialogue": dialogue
        })
    
    return processed_data

DATASET_PROCESSOR_MAP = {
    "cider": process_cider,
    "wow": process_wow,
    "cmsd": process_cmsd,
    "reddit": process_reddit,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data.", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name.", required=True)
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to generate.")
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()
    processed_data = DATASET_PROCESSOR_MAP[args.dataset](args.datapath, args.num_samples)
    mturk_data = []

    for sample in tqdm(processed_data, desc="Preparing data"):
        mturk_data.append({
            "id": sample["id"],
            "dialogue": mturk_prepare_dialogue(sample["dialogue"], include_index=True),
            "num_turns": len(sample["dialogue"])
        })
    
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_odd_cka{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()