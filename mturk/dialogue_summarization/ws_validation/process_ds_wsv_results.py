import sys
import argparse
import pandas as pd
import pathlib
import json
import math
from tqdm import tqdm

sys.path.append("../../..")

from utils import write_json, mturk_not_empty, mturk_extract_size, mturk_process_dialogue, mturk_process_json, mturk_process_dialogue_turn, clean_utterance

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
            "dialogue": mturk_process_dialogue(row["Input.dialogue"]),
            "summary": row["Input.summary"],
            "knowledge": [kg for kg in mturk_process_json(row["Input.knowledge"])],
            "alt_targets": mturk_process_json(row["Input.ws_options"]),
            "new_alt_targets": [],
            "option_none": False,
            "bonus": 0
        }

        feedback = row["Answer.feedback"]
    
        if mturk_not_empty(feedback):
            result["worker_feedback"] = feedback.strip()

        num_summaries = mturk_extract_size(results, "Answer.alt-summary<num>")
        num_options = mturk_extract_size(results, "Answer.ws-option<num>.*")

        for o_index in range(1, num_options+1):
            option = row[f"Answer.ws-option{o_index}"]
            if mturk_not_empty(option):
                for summary in result["alt_targets"]:
                    if summary["id"] == option:
                        summary["selected"] = True
                        break
        
        option_none = row["Answer.ws-option-none"]

        if mturk_not_empty(option_none):
            result["option_none"] = False if math.isnan(option_none) else int(option_none) == 1

        # justification = row["Answer.justification"]

        # if mturk_not_empty(justification):
        #     result["justification"] = justification

        for s_index in range(1, num_summaries+1):
            alt_summary_text = row[f"Answer.alt-summary{s_index}"].strip()

            if mturk_not_empty(alt_summary_text):
                alt_summary = {
                    "id": f"{result['assignment_id']}-{s_index-1}",
                    "text": alt_summary_text
                }
                
                result["new_alt_targets"].append(alt_summary)

        result["bonus"] = len(result["new_alt_targets"])-1

        if result["bonus"] > 0:
            result["bonus_reason"] = "Bonus for additional annotations."

        json_results.append(result)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    write_json(json_results, json_results_path)

if __name__ == "__main__":
    main()