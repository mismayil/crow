import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, escape_quotes, write_json

def prepare_chinese_mt(datapath):
    data = read_json(datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        mturk_sample = {
            "id": sample["data_id"],
            "plausible_sentence": sample["plausible_sentence"],
            "implausible_sentence": sample["implausible_sentence"],
            "ck_options": []
        }
        
        kg_list = []

        for _, kg in enumerate(sample["knowledge"]):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])

            if kg_tuple not in kg_list:
                mturk_sample["ck_options"].append({
                    "id": kg["id"], 
                    "head": escape_quotes(kg["head"]), 
                    "relation": kg["relation"], 
                    "tail": escape_quotes(kg["tail"])
                })
                kg_list.append(kg_tuple)
        
        mturk_data.append(mturk_sample)
    
    return mturk_data

def highlight(sentence, word, highlight_class="highlight"):
    words = sentence.split(" ")

    for i in range(len(words)-1, 0, -1):
        if words[i].lower().strip() == word.lower().strip():
            words[i] = f'<strong class="{highlight_class}">{words[i]}</strong>'
            break
    
    return " ".join(words)

def prepare_wino_x(datapath):
    data = read_json(datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        sentence = highlight(sample["sentence"], "it")
        true_referent = sample["true_referent"]
        false_referent = sample["false_referent"]
        sentence = sentence.replace(true_referent, f'<strong class="concept_a">{true_referent}</strong>', 1)
        sentence = sentence.replace(false_referent, f'<strong class="concept_b">{false_referent}</strong>', 1)
        
        mturk_sample = {
            "id": sample["data_id"],
            "sentence": sentence,
            "true_referent": true_referent,
            "false_referent": false_referent,
            "ck_options": []
        }
        
        kg_list = []

        for _, kg in enumerate(sample["knowledge"]):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])

            if kg_tuple not in kg_list:
                mturk_sample["ck_options"].append({
                    "id": kg["id"], 
                    "head": escape_quotes(kg["head"]), 
                    "relation": kg["relation"], 
                    "tail": escape_quotes(kg["tail"])
                })
                kg_list.append(kg_tuple)
        
        mturk_data.append(mturk_sample)
    
    return mturk_data

DATASET_PROCESSOR_MAP = {
    "chinese_mt": prepare_chinese_mt,
    "wino_x": prepare_wino_x
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated json data (results of CKA step).", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset name.", required=True, choices=DATASET_PROCESSOR_MAP.keys())
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()

    mturk_data = DATASET_PROCESSOR_MAP[args.dataset](args.datapath)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_mt_ckv{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")


if __name__ == "__main__":
    main()