import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, clean, write_json, mturk_prepare_dialogue, text_to_wordset, chunk

def process_dialogsum(datapath, num_samples=0):
    data = read_json(datapath)

    if num_samples > 0:
        data = random.sample(data, num_samples)
    
    processed_data = []

    for sample in tqdm(data, desc="Processing data"):
        if "summaries" in sample:
            for summary in sample["summaries"]:
                processed_data.append({
                    "id": sample["id"],
                    "dialogue": sample["dialogue"],
                    "summary": clean(summary["text"])
                })
        else:
            processed_data.append({
                "id": sample["id"],
                "dialogue": sample["dialogue"],
                "summary": clean(sample["summary"]["text"])
            })
    
    return processed_data

def process_samsum(datapath, num_samples=0):
    return process_dialogsum(datapath, num_samples)

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
            "dialogue": dialogue,
            "summary": sample["summary"]
        })
    
    return processed_data

DATASET_PROCESSOR_MAP = {
    "dialogsum": process_dialogsum,
    "samsum": process_samsum,
    "reddit": process_reddit
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data.", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name.", required=True)
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to generate.")
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory.")
    parser.add_argument("--summary-length-threshold", type=int, default=5, help="Summary length threshold.")
    parser.add_argument("--dialogue-length-threshold", type=int, default=4, help="Dialogue length threshold.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data.")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size.")

    args = parser.parse_args()
    processed_data = DATASET_PROCESSOR_MAP[args.dataset](args.datapath, args.num_samples)
    mturk_data = []

    for sample in tqdm(processed_data, desc="Preparing data"):
        summary_wordset = text_to_wordset(sample["summary"], ignore_stopwords=True)

        if len(summary_wordset) >= args.summary_length_threshold and len(sample["dialogue"]) >= args.dialogue_length_threshold:
            mturk_data.append({
                "id": sample["id"],
                "dialogue": mturk_prepare_dialogue(sample["dialogue"], include_index=True),
                "summary": sample["summary"]
            })
    
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    if args.batch_size > 0:
        if args.shuffle:
            random.shuffle(mturk_data)
        mturk_data_batches = chunk(mturk_data, args.batch_size)
        for index, batch in enumerate(mturk_data_batches):
            pd.DataFrame(batch).to_csv(f"{args.output_dir}/mturk_data_ds_cka{args.suffix}_batch{index+1}.csv", index=False)
    else:
        pd.DataFrame(mturk_data).to_csv(f"{args.output_dir}/mturk_data_ds_cka{args.suffix}.csv", index=False)
    
    write_json(mturk_data, f"{args.output_dir}/mturk_data_ds_cka{args.suffix}.json")

if __name__ == "__main__":
    main()