import sys
import argparse
import pandas as pd
import pathlib
import json
import math
from tqdm import tqdm

sys.path.append("../../..")

from utils import write_json, mturk_not_empty, mturk_extract_size, mturk_process_json, mturk_process_text, clean_utterance

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
            "headline": row["Input.headline"],
            "intent": row["Input.intent"],
            "intent_id": row["Input.intent_id"],
            "knowledge": [kg for kg in mturk_process_json(row["Input.knowledge"])],
            "alt_targets": [{**target, "text": mturk_process_text(target["text"])} for target in mturk_process_json(row["Input.ws_options"])],
            "new_alt_targets": [],
            "option_none": False,
            "bonus": 0
        }

        feedback = row["Answer.feedback"]
    
        if mturk_not_empty(feedback):
            result["worker_feedback"] = feedback.strip()

        num_targets = mturk_extract_size(results, "Answer.alt-target<num>")
        num_options = mturk_extract_size(results, "Answer.ws-option<num>.*")

        for o_index in range(1, num_options+1):
            option = row[f"Answer.ws-option{o_index}"]
            if mturk_not_empty(option):
                for target in result["alt_targets"]:
                    if target["id"] == option:
                        target["selected"] = True
                        break
        
        option_none = row["Answer.ws-option-none"]

        if mturk_not_empty(option_none):
            result["option_none"] = False if math.isnan(option_none) else int(option_none) == 1

        justification = row.get("Answer.justification")

        if mturk_not_empty(justification):
            result["justification"] = justification

        for t_index in range(1, num_targets+1):
            alt_target_text = row[f"Answer.alt-target{t_index}"].strip()
            alt_target_ck = row.get(f"Answer.alt-target{t_index}-ck")

            if mturk_not_empty(alt_target_text) and mturk_not_empty(alt_target_ck):
                alt_target = {
                    "id": f"{result['assignment_id']}-{t_index-1}",
                    "text": clean_utterance(alt_target_text),
                    "knowledge_ids": alt_target_ck.split(",")
                }
                
                result["new_alt_targets"].append(alt_target)

        result["bonus"] = len(result["new_alt_targets"])-1

        if result["bonus"] > 0:
            result["bonus_reason"] = "Bonus for additional annotations."

        json_results.append(result)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    write_json(json_results, json_results_path)

if __name__ == "__main__":
    main()