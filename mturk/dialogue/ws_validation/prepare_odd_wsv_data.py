import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, escape_quotes, mturk_prepare_dialogue, mturk_prepare_dialogue_turn, clean_utterance, parse_turn, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to aggregated json data (results of WSG step).", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for processed data.")

    args = parser.parse_args()
    data = read_json(args.datapath)

    mturk_data = []

    for sample in tqdm(data, desc="Preparing data", total=len(data)):
        if not sample["no_alt_target"] and sample["dialogue"]:
            final_turn_speaker, final_turn_utterance = parse_turn(sample["final_turn"])
            mturk_sample = {
                "id": sample["data_id"],
                "dialogue": mturk_prepare_dialogue(sample["dialogue"]),
                "final_turn": mturk_prepare_dialogue_turn(final_turn_speaker, final_turn_utterance),
                "final_turn_prefix": sample["final_turn_prefix"],
                "knowledge": [{"id": kg["id"], "head": escape_quotes(kg["head"]), "relation": kg["relation"], "tail": escape_quotes(kg["tail"]), "head_turn": kg["head_turn"], "tail_turn": kg["tail_turn"]} for kg in sample["knowledge"]],
                "ws_options": []
            }
            
            turn_list = []

            for index, turn in enumerate(sample["alt_targets"]):
                if turn["text"] not in turn_list:
                    turn_list.append(turn["text"])
                    mturk_sample["ws_options"].append({
                        "id": turn["id"], 
                        "text": f'{sample["final_turn_prefix"]} {clean_utterance(escape_quotes(turn["text"]))}'
                    })
            
            mturk_data.append(mturk_sample)

    output_path_stem = f"{args.output_dir}/mturk_data_odd_wsv{args.suffix}"
    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    pd.DataFrame(mturk_data).to_csv(f"{output_path_stem}.csv", index=False)
    write_json(mturk_data, f"{output_path_stem}.json")

if __name__ == "__main__":
    main()