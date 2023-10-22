import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../../..")

from utils import read_json, write_json, text_to_wordset, escape_quotes

def process_mrf(datapath):
    data = read_json(datapath)
    
    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        headline_wordset = text_to_wordset(sample["headline"], ignore_stopwords=True)

        if len(headline_wordset) >= 5:
            intents = []

            for intent in sample["writer_intent"]:
                intent_wordset = text_to_wordset(intent, ignore_stopwords=True)

                if len(intent_wordset) >= 5:
                    intents.append(intent.strip('"').strip("'"))

            if len(intents) > 0:
                mturk_data.append({
                    "id": sample["id"],
                    "headline": sample["headline"],
                    "intents": [{"id": f"{sample['id']}-intent-{i_index}", "text": escape_quotes(intent)} for i_index, intent in enumerate(intents)]
                })
    
    return mturk_data

DATASET_PROCESSOR_MAP = {
    "mrf": process_mrf
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
    output_path_stem = f"{args.output_dir}/mturk_data_id_cka{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()