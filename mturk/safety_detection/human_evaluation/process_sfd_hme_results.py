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
            "scenario": row["Input.scenario"],
            "action": row["Input.action"],
            "action_id": row["Input.action_id"],
            "label": row["Input.label"],
            "answer": row["Answer.answer"],
        }

        feedback = row["Answer.feedback"]
    
        if mturk_not_empty(feedback):
            result["worker_feedback"] = feedback.strip()

        json_results.append(result)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    write_json(json_results, json_results_path)

if __name__ == "__main__":
    main()