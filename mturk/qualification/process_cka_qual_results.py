import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import re
import math

sys.path.append("../..")

from utils import write_json, mturk_convert_prefix, mturk_process_dialogue, mturk_not_empty

MTURK_DIALOGUE_TEMPLATE = "<strong>{index}. {speaker}:</strong> {utterance}<br>"

def get_num_questions_options(results, pattern, q_prefix="q", opt_prefix="opt"):
    num_questions = []
    num_options = []
    pattern = pattern.replace(f"{q_prefix}<num>", f"{q_prefix}(?P<q_num>\d+)").replace(f"{opt_prefix}<num>", f"{opt_prefix}(?P<opt_num>\d+)")

    for col in results.columns:
        match = re.fullmatch(pattern, col)
        q_num = match.group("q_num") if match else None
        opt_num = match.group("opt_num") if match else None
        if q_num:
            num_questions.append(int(q_num))
        if opt_num:
            num_options.append(int(opt_num))
    
    max_num_questions = max(num_questions) if num_questions else 0
    max_num_options = max(num_options) if num_options else 0

    return max_num_questions, max_num_options

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to results.", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")

    args = parser.parse_args()
    results = pd.read_csv(args.results_path)
    json_data = []

    mcq_odd_num_questions, mcq_odd_num_options = get_num_questions_options(results, "Input.mcq_odd_q<num>_opt<num>_.*")
    mcq_ds_num_questions, mcq_ds_num_options = get_num_questions_options(results, "Input.mcq_ds_q<num>_opt<num>_.*")
    mcq_mt_num_questions, mcq_mt_num_options = get_num_questions_options(results, "Input.mcq_mt_q<num>_opt<num>_.*")

    for r_index, row in tqdm(results.iterrows(), total=len(results), desc="Processing results"):
        json_sample = {
            "data_id": row["Input.id"],
            "hit_id": row["HITId"],
            "assignment_id": row["AssignmentId"],
            "worker_id": row["WorkerId"],
            "worker_time": float(row["WorkTimeInSeconds"]),
            "worker_ee": float(row["Answer.ee"] if mturk_not_empty(row["Answer.ee"]) else row["WorkTimeInSeconds"]),
            "feedback": row["Answer.feedback"]
        }

        json_sample["mcq_odd_questions"] = []

        for q_index in range(1, mcq_odd_num_questions+1):
            question = {
                "dialogue": mturk_process_dialogue(row[f"Input.mcq_odd_q{q_index}_dialogue"]),
                "options": []
            }

            for o_index in range(1, mcq_odd_num_options+1):
                option = {
                    "head": row[f"Input.mcq_odd_q{q_index}_opt{o_index}_head"],
                    "relation": row[f"Input.mcq_odd_q{q_index}_opt{o_index}_rel"],
                    "tail": row[f"Input.mcq_odd_q{q_index}_opt{o_index}_tail"],
                    "label": row[f"Input.mcq_odd_q{q_index}_opt{o_index}_label"],
                    "answer": 0
                }

                answer_col = f"Answer.mcq-odd-q{q_index}-opt{o_index}"

                if answer_col in row:
                    option["answer"] = 0 if math.isnan(row[answer_col]) else int(row[answer_col])

                question["options"].append(option)
            
            json_sample["mcq_odd_questions"].append(question)
        
        json_sample["mcq_ds_questions"] = []

        for q_index in range(1, mcq_ds_num_questions+1):
            question = {
                "dialogue": mturk_process_dialogue(row[f"Input.mcq_ds_q{q_index}_dialogue"]),
                "summary": row[f"Input.mcq_ds_q{q_index}_summary"],
                "options": []
            }

            for o_index in range(1, mcq_ds_num_options+1):
                option = {
                    "head": row[f"Input.mcq_ds_q{q_index}_opt{o_index}_head"],
                    "relation": row[f"Input.mcq_ds_q{q_index}_opt{o_index}_rel"],
                    "tail": row[f"Input.mcq_ds_q{q_index}_opt{o_index}_tail"],
                    "label": row[f"Input.mcq_ds_q{q_index}_opt{o_index}_label"],
                    "answer": 0
                }

                answer_col = f"Answer.mcq-ds-q{q_index}-opt{o_index}"

                if answer_col in row:
                    option["answer"] = 0 if math.isnan(row[answer_col]) else int(row[answer_col])

                question["options"].append(option)
            
            json_sample["mcq_ds_questions"].append(question)
        
        json_sample["mcq_mt_questions"] = []

        for q_index in range(1, mcq_mt_num_questions+1):
            question = {
                "sentence1": row[f"Input.mcq_mt_q{q_index}_sentence1"],
                "sentence2": row[f"Input.mcq_mt_q{q_index}_sentence2"],
                "implausible_label": row[f"Input.mcq_mt_q{q_index}_implausible_label"],
                "implausible_answer": row[f"Answer.mcq-mt-q{q_index}-sent"],
                "options": []
            }

            ck_answer = row[f"Answer.mcq-mt-q{q_index}-ck"]

            for o_index in range(1, mcq_mt_num_options+1):
                option = {
                    "head": row[f"Input.mcq_mt_q{q_index}_opt{o_index}_head"],
                    "relation": row[f"Input.mcq_mt_q{q_index}_opt{o_index}_rel"],
                    "tail": row[f"Input.mcq_mt_q{q_index}_opt{o_index}_tail"],
                    "label": row[f"Input.mcq_mt_q{q_index}_opt{o_index}_label"],
                    "answer": 0
                }

                if ck_answer == o_index:
                    option["answer"] = 1

                question["options"].append(option)
            
            json_sample["mcq_mt_questions"].append(question)
        
        json_sample["oq_odd_questions"] = []
        oq_odd_num_questions, oq_odd_num_kg = get_num_questions_options(results, "Answer.oq-odd-q<num>-ck<num>-.*", opt_prefix="ck")

        for q_index in range(1, oq_odd_num_questions+1):
            question = {
                "dialogue": mturk_process_dialogue(row[f"Input.oq_odd_q{q_index}_dialogue"]),
                "num_turns": row[f"Input.oq_odd_q{q_index}_num_turns"]
            }

            knowledge = []

            for k_index in range(1, oq_odd_num_kg+1):
                kg = {}
                head_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-head"
                head_turn_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-head-turn"
                rel_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-relation"
                tail_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-tail"
                tail_turn_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-tail-turn"
                rel_prefix_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-relation-prefix"
                rel_other_col = f"Answer.oq-odd-q{q_index}-ck{k_index}-relation-other"

                relation_prefix = mturk_convert_prefix(row[rel_prefix_col] if rel_prefix_col in row else "")
                relation = row[rel_col] if rel_col in row else ""
                relation_other = row[rel_other_col] if rel_other_col in row else ""
                head = row[head_col] if head_col in row else ""
                tail = row[tail_col] if tail_col in row else ""
                head_turn = row[head_turn_col] if head_turn_col in row else ""
                tail_turn = row[tail_turn_col] if tail_turn_col in row else ""

                if mturk_not_empty(head) and mturk_not_empty(relation) and mturk_not_empty(tail):
                    kg["head"] = head
                    kg["relation"] = f"{relation_prefix} {relation}".strip()
                    kg["tail"] = tail

                    if mturk_not_empty(relation_other):
                        kg["relation_other"] = relation_other

                    if mturk_not_empty(head_turn):
                        kg["head_turn"] = head_turn
                    
                    if mturk_not_empty(tail_turn):
                        kg["tail_turn"] = tail_turn

                    knowledge.append(kg)
            
            question["knowledge"] = knowledge
            json_sample["oq_odd_questions"].append(question)

        json_data.append(json_sample)
    
    results_path = pathlib.Path(args.results_path)
    json_results_path = results_path.parent / f"{results_path.stem}.json"

    write_json(json_data, json_results_path)

if __name__ == "__main__":
    main()