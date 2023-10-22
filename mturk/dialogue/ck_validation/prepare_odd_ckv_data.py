import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, escape_quotes, dim_from_relation, mturk_prepare_dialogue, write_json

def prepare_kg_for_mturk(kg, kg_id):
    head_turn = kg.get("head_turn", kg.get("head_index", -1)+1)
    tail_turn = kg.get("tail_turn", kg.get("tail_index", -1)+1)

    return {
        "id": kg_id,
        "head": escape_quotes(kg["head"]),
        "relation": kg["relation"],
        "tail": escape_quotes(kg["tail"]),
        "head_turn": head_turn,
        "tail_turn": tail_turn
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated json data (results of CKA step).", required=True)
    parser.add_argument("--cider-datapath", type=str, help="Path to original CIDER json data.")
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    cider_data_map = {}

    if args.cider_datapath is not None:
        cider_data = read_json(args.cider_datapath)
        cider_data_map = {sample["id"]: sample for sample in cider_data}

    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        mturk_sample = {
            "id": sample["data_id"],
            "dialogue": mturk_prepare_dialogue(sample["dialogue"]),
            "num_turns": len(sample["dialogue"]),
            "ck_options": []
        }

        kg_list = []
        cider_sample = cider_data_map.get(sample["data_id"], None)

        if cider_sample:
            for index, kg in enumerate(cider_sample["triplets"]):
                dim = dim_from_relation(kg["relation"], ignore_other=True)
                if dim:
                    kg_tuple = (kg["head"], kg["relation"], kg["tail"])
                    if kg_tuple not in kg_list:
                        mturk_sample["ck_options"].append(prepare_kg_for_mturk(kg, f'cider-{cider_sample["id"]}-{index}'))
                        kg_list.append(kg_tuple)
        
        for _, kg in enumerate(sample["knowledge"]):
            kg_tuple = (kg["head"], kg["relation"], kg["tail"])
            
            if kg_tuple not in kg_list:
                mturk_sample["ck_options"].append(prepare_kg_for_mturk(kg, kg["id"]))
                kg_list.append(kg_tuple)

        mturk_data.append(mturk_sample)

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    output_path_stem = f"{args.output_dir}/mturk_data_odd_ckv{args.suffix}"
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()