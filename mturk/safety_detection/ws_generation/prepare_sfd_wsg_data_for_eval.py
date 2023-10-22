import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to results in json (results of WSG step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")

    args = parser.parse_args()
    results = read_json(args.results_path)

    eval_data = {}

    for result in tqdm(results, desc="Preparing results for evaluation"):
        eval_result = eval_data.get(result["data_id"])

        if not eval_result:
            eval_result = {
                "data_id": result["data_id"],
                "scenario": result["scenario"],
                "actions": []
            }

            for action in result["safe_actions"]:
                eval_result["actions"].append({
                    **action,
                    "label": "safe"
                })
            
            for action in result["unsafe_actions"]:
                eval_result["actions"].append({
                    **action,
                    "label": "unsafe"
                })
            
            eval_data[result["data_id"]] = eval_result

        if eval_result and not result["no_alt_target"]:
            for alt_target in result["alt_targets"]:
                selected_knowledge = [kg for kg in result["knowledge"] if kg["id"] in alt_target["knowledge_ids"]]
                action_id = alt_target["action_id"]
                eval_result["actions"].append({
                    "id": alt_target["id"],
                    "text": alt_target["text"],
                    "knowledge": selected_knowledge,
                    "action_id": action_id,
                    "label": "unsafe" if any([action["id"] == action_id and action["label"] == "safe" for action in eval_result["actions"]]) else "safe"
                })
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()