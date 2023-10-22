import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, write_json

def process_chinese_mt(datapath):
    data = read_json(datapath)
    
    sentence_pairs = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        sentence_pairs.append({
            "id": sample["id"],
            "plausible_sentence": sample["english_target_correct"],
            "implausible_sentence": sample["english_target_wrong"]
        })
    
    return sentence_pairs

def highlight(sentence, word, highlight_class="highlight"):
    words = sentence.split(" ")

    for i in range(len(words)-1, 0, -1):
        if words[i].lower().strip() == word.lower().strip():
            words[i] = f'<strong class="{highlight_class}">{words[i]}</strong>'
            break
    
    return " ".join(words)

def process_wino_x(datapath):
    data = read_json(datapath)
    
    sentence_pairs = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        sentence = highlight(sample["sentence"], "it")
        true_referent = sample[f"referent{sample['answer']}_en"]
        false_referent = sample[f"referent{3-sample['answer']}_en"]
        sentence = sentence.replace(true_referent, f'<strong class="concept_a">{true_referent}</strong>', 1)
        sentence = sentence.replace(false_referent, f'<strong class="concept_b">{false_referent}</strong>', 1)

        sentence_pairs.append({
            "id": sample["qID"],
            "sentence": sentence,
            "true_referent": true_referent,
            "false_referent": false_referent
        })
    
    return sentence_pairs

DATASET_PROCESSOR_MAP = {
    "chinese_mt": process_chinese_mt,
    "wino_x": process_wino_x
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
    output_path_stem = f"{args.output_dir}/mturk_data_mt_cka{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()