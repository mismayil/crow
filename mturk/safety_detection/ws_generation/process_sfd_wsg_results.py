import sys
import argparse
import pandas as pd
import pathlib
import math
from tqdm import tqdm

sys.path.append("../../..")

from utils import write_json, mturk_not_empty, mturk_extract_size, mturk_process_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to csv results.", required=True)

    args = parser.parse_args()
    results = pd.read_csv(args.results_path)
    json_results = []

    for index, row in tqdm(results.iterrows(), desc="Processing results", total=len(results)):
        result = {
            "data_id": row["Input.id"],
            "hit_id": row["HITId"],
            "assignment_id": row["AssignmentId"],
            "worker_id": row["WorkerId"],
            "worker_time": row["WorkTimeInSeconds"],
            "worker_ee": row["Answer.ee"],
            "scenario": row["Input.scenario"],
            "safe_actions": mturk_process_json(row["Input.safe_actions"]),
            "unsafe_actions": mturk_process_json(row["Input.unsafe_actions"]),
            "knowledge": mturk_process_json(row["Input.knowledge"]),
            "alt_targets": [],
            "no_alt_target": False,
            "bonus": 0
        }

        feedback = row["Answer.feedback"]
    
        if mturk_not_empty(feedback):
            result["worker_feedback"] = feedback.strip()

        num_alt_targets = mturk_extract_size(results, "Answer.alt-target<num>")

        for t_index in range(1, num_alt_targets+1):
            alt_target_text = row.get(f"Answer.alt-target{t_index}")
            alt_target_ck = row.get(f"Answer.alt-target{t_index}-ck")
            alt_target_action = row.get(f"Answer.alt-target{t_index}-action")

            if mturk_not_empty(alt_target_text) and mturk_not_empty(alt_target_ck) and mturk_not_empty(alt_target_action):
                alt_target = {
                    "id": f'{result["assignment_id"]}-{t_index-1}',
                    "text": alt_target_text,
                    "knowledge_ids": alt_target_ck.split(","),
                    "action_id": alt_target_action
                }

                result["alt_targets"].append(alt_target)
        
        option_none = row.get("Answer.no-alt-target")

        if mturk_not_empty(option_none):
            result["no_alt_target"] = False if math.isnan(option_none) else int(option_none) == 1

        option_none_desc = row.get("Answer.no-alt-target-feedback")

        if mturk_not_empty(option_none_desc):
            result["no_alt_target_desc"] = option_none_desc

        result["bonus"] = len(result["alt_targets"])-1

        if result["bonus"] > 0:
            result["bonus_reason"] = "Bonus for additional annotations."

        json_results.append(result)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    write_json(json_results, json_results_path)

if __name__ == "__main__":
    main()