import sys
import argparse
import pandas as pd
import pathlib
import json
import math
from tqdm import tqdm

sys.path.append("../../..")

from utils import mturk_convert_prefix, mturk_not_empty, mturk_extract_size, mturk_process_json, mturk_process_text

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
            "knowledge": mturk_process_json(row["Input.ck_options"]),
            "new_knowledge": [],
            "option_none": False,
            "bonus": 0
        }

        sentence = row.get("Input.sentence")

        if sentence:
            true_referent = row["Input.true_referent"]
            false_referent = row["Input.false_referent"]
            result["sentence"] = mturk_process_text(sentence)
            result["true_referent"] = true_referent
            result["false_referent"] = false_referent
        else:
            result["plausible_sentence"] = mturk_process_text(row["Input.plausible_sentence"])
            result["implausible_sentence"] = mturk_process_text(row["Input.implausible_sentence"])

        feedback = row["Answer.feedback"]
    
        if mturk_not_empty(feedback):
            result["worker_feedback"] = feedback.strip()

        num_knowledge = mturk_extract_size(results, "Answer.ck<num>-.*")
        num_options = mturk_extract_size(results, "Answer.ck-option<num>.*")

        for o_index in range(1, num_options+1):
            option = row[f"Answer.ck-option{o_index}"]
            if mturk_not_empty(option):
                for kg in result["knowledge"]:
                    if kg["id"] == option:
                        kg["selected"] = True
                        break
        
        option_none = row["Answer.ck-option-none"]

        if mturk_not_empty(option_none):
            result["option_none"] = False if math.isnan(option_none) else int(option_none) == 1

        for k_index in range(1, num_knowledge+1):
            head = row[f"Answer.ck{k_index}-head"].strip()
            relation  = row[f"Answer.ck{k_index}-relation"]
            tail = row[f"Answer.ck{k_index}-tail"].strip()
            prefix = mturk_convert_prefix(row[f"Answer.ck{k_index}-relation-prefix"])
            other_relation = row[f"Answer.ck{k_index}-relation-other"].strip()

            if mturk_not_empty(head) and mturk_not_empty(tail) and mturk_not_empty(relation):
                kg = {
                    "id": f"{result['assignment_id']}-{k_index-1}",
                    "head": head,
                    "relation": f"{prefix} {relation}".strip(),
                    "tail": tail
                }

                if mturk_not_empty(other_relation):
                    kg["other_relation"] = other_relation
                
                result["new_knowledge"].append(kg)

        result["bonus"] = len(result["new_knowledge"])-1

        if result["bonus"] > 0:
            result["bonus_reason"] = "Bonus for additional annotations."

        json_results.append(result)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    with open(json_results_path, "w") as f:
        json.dump(json_results, f, indent=4)

if __name__ == "__main__":
    main()