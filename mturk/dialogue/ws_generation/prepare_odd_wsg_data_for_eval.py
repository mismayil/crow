import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, prepare_dialogue_turn_for_eval, prepare_dialogue_for_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to results in json (results of WSG step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")

    args = parser.parse_args()
    results = read_json(args.results_path)

    eval_data = {}

    for result in tqdm(results, desc="Preparing results for evaluation"):
        eval_result = eval_data.get(result["hit_id"])

        if not eval_result:
            if result["dialogue"]:
                eval_result = {
                    "data_id": result["data_id"],
                    "hit_id": result["hit_id"],
                    "dialogue": prepare_dialogue_for_eval(result["dialogue"]),
                    "knowledge": result["knowledge"],
                    "final_turns": [{
                        "id": result["assignment_id"],
                        "text": prepare_dialogue_turn_for_eval(result["final_turn"]),
                        "label": 1
                    }]
                }
                eval_data[result["hit_id"]] = eval_result

        if eval_result and not result["no_alt_target"]:
            for alt_turn in result["alt_targets"]:
                eval_result["final_turns"].append({
                    "id": result["assignment_id"],
                    "text": prepare_dialogue_turn_for_eval(alt_turn), 
                    "label": 0
                })
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()