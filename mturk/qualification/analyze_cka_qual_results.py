import sys
import argparse
from tqdm import tqdm
import numpy as np
import pathlib
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append("../..")

from utils import read_json, get_dim_distribution, write_json

def add_stats_by_qtype(report, results, qtype):
    num_workers = results.shape[0]
    num_questions = results.shape[1]

    for w_index in tqdm(range(num_workers), desc=f"Collecting {qtype} stats per worker"):
        worker = report["workers"][w_index]

        for q_index in range(num_questions):
            q_res = results[w_index, q_index]
            q_accuracy = round(accuracy_score(q_res[:, 0], q_res[:, 1]), 2)
            q_f1 = round(f1_score(q_res[:, 0], q_res[:, 1]), 2)
            q_precision = round(precision_score(q_res[:, 0], q_res[:, 1]), 2)
            q_recall = round(recall_score(q_res[:, 0], q_res[:, 1]), 2)
            worker["questions"][qtype]["q_stats"].append({'accuracy': q_accuracy, 'f1': q_f1, 'precision': q_precision, 'recall': q_recall})
        
        worker["questions"][qtype]["accuracy"] = round(accuracy_score(results[w_index, :, :, 0].flatten(), results[w_index, :, :, 1].flatten()), 2)
        worker["questions"][qtype]["f1"] = round(f1_score(results[w_index, :, :, 0].flatten(), results[w_index, :, :, 1].flatten()), 2)
        worker["questions"][qtype]["precision"] = round(precision_score(results[w_index, :, :, 0].flatten(), results[w_index, :, :, 1].flatten()), 2)
        worker["questions"][qtype]["recall"] = round(recall_score(results[w_index, :, :, 0].flatten(), results[w_index, :, :, 1].flatten()), 2)

    for q_index in tqdm(range(num_questions), desc=f"Collecting {qtype} stats per question"):
        q_res = results[:, q_index]
        q_accuracy = round(accuracy_score(q_res[:, :, 0].flatten(), q_res[:, :, 1].flatten()), 2)
        q_f1 = round(f1_score(q_res[:, :, 0].flatten(), q_res[:, :, 1].flatten()), 2)
        q_precision = round(precision_score(q_res[:, :, 0].flatten(), q_res[:, :, 1].flatten()), 2)
        q_recall = round(recall_score(q_res[:, :, 0].flatten(), q_res[:, :, 1].flatten()), 2)
        report["questions"][qtype]["q_stats"].append({'accuracy': q_accuracy, 'f1': q_f1, 'precision': q_precision, 'recall': q_recall})
    
    report["questions"][qtype]["accuracy"] = round(accuracy_score(results[:, :, :, 0].flatten(), results[:, :, :, 1].flatten()), 2)
    report["questions"][qtype]["f1"] = round(f1_score(results[:, :, :, 0].flatten(), results[:, :, :, 1].flatten()), 2)
    report["questions"][qtype]["precision"] = round(precision_score(results[:, :, :, 0].flatten(), results[:, :, :, 1].flatten()), 2)
    report["questions"][qtype]["recall"] = round(recall_score(results[:, :, :, 0].flatten(), results[:, :, :, 1].flatten()), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to json results.", required=True)
    args = parser.parse_args()

    results = read_json(args.results_path)
    work_times = [res["worker_time"] for res in results]
    ee_times = [res["worker_ee"] for res in results]

    num_workers = len(results)
    num_mcq_odd_questions = len(results[0]["mcq_odd_questions"])
    num_mcq_ds_questions = len(results[0]["mcq_ds_questions"])
    num_mcq_mt_questions = len(results[0]["mcq_mt_questions"])
    num_mcq_odd_options = len(results[0]["mcq_odd_questions"][0]["options"])
    num_mcq_ds_options = len(results[0]["mcq_ds_questions"][0]["options"])
    num_mcq_mt_options = len(results[0]["mcq_mt_questions"][0]["options"])+2
    mcq_odd_results = np.zeros((num_workers, num_mcq_odd_questions, num_mcq_odd_options, 2))
    mcq_ds_results = np.zeros((num_workers, num_mcq_ds_questions, num_mcq_ds_options, 2))
    mcq_mt_results = np.zeros((num_workers, num_mcq_mt_questions, num_mcq_mt_options, 2))
    knowledge = []

    report = {
        "mean_work_time": sum(work_times) / len(results),
        "mean_ee_time": sum(ee_times) / len(results),
        "max_work_time": max(work_times),
        "max_ee_time": max(ee_times),
        "min_work_time": min(work_times),
        "min_ee_time": min(ee_times),
        "workers": [],
        "questions": {
            "mcq_odd": {
                "num_questions": num_mcq_odd_questions,
                "num_options": num_mcq_odd_options,
                "accuracy": "",
                "f1": "", 
                "precision": "", 
                "recall": "", 
                "q_stats": []
            },
            "mcq_ds": {
                "num_questions": num_mcq_ds_questions,
                "num_options": num_mcq_ds_options,
                "accuracy": "",
                "f1": "", 
                "precision": "", 
                "recall": "", 
                "q_stats": []
            },
            "mcq_mt": {
                "num_questions": num_mcq_mt_questions,
                "num_options": num_mcq_mt_options,
                "accuracy": "",
                "f1": "", 
                "precision": "", 
                "recall": "", 
                "q_stats": []
            }
        }
    }

    head_rel_tail_map = defaultdict(list)
    head_tail_map = defaultdict(list)

    for w_index, result in tqdm(enumerate(results), total=len(results), desc="Indexing results"):
        worker = {
            "worker_id": result["worker_id"],
            "hit_id": result["hit_id"],
            "assignment_id": result["assignment_id"],
            "work_time": result["worker_time"],
            "ee_time": result["worker_ee"],
            "accuracy": "", 
            "f1": "", 
            "precision": "",
            "recall": "",
            "questions": {
                "mcq_odd": {"accuracy": "", "f1": "", "precision": "", "recall": "", "q_stats": []}, 
                "mcq_ds": {"accuracy": "", "f1": "", "precision": "", "recall": "", "q_stats": []}, 
                "mcq_mt": {"accuracy": "", "f1": "", "precision": "", "recall": "", "q_stats": []}
            }
        }

        for q_index, question in enumerate(result["mcq_odd_questions"]):
            for o_index, opt in enumerate(question["options"]):
                mcq_odd_results[w_index, q_index, o_index, 0] = opt["label"]
                mcq_odd_results[w_index, q_index, o_index, 1] = opt["answer"]

        for q_index, question in enumerate(result["mcq_ds_questions"]):
            for o_index, opt in enumerate(question["options"]):
                mcq_ds_results[w_index, q_index, o_index, 0] = opt["label"]
                mcq_ds_results[w_index, q_index, o_index, 1] = opt["answer"]

        for q_index, question in enumerate(result["mcq_mt_questions"]):
            mcq_mt_results[w_index, q_index, 0, 0] = int(question["implausible_label"] == 1)
            mcq_mt_results[w_index, q_index, 0, 1] = int(question["implausible_answer"] == 1)
            mcq_mt_results[w_index, q_index, 1, 0] = int(question["implausible_label"] == 2)
            mcq_mt_results[w_index, q_index, 1, 1] = int(question["implausible_answer"] == 2)

            for o_index, opt in enumerate(question["options"]):
                mcq_mt_results[w_index, q_index, o_index+2, 0] = opt["label"]
                mcq_mt_results[w_index, q_index, o_index+2, 1] = opt["answer"]

        for question in result["oq_odd_questions"]:
            knowledge.extend(question["knowledge"])
            for kg in question["knowledge"]:
                head_rel_tail_map[(kg["head"], kg["relation"], kg["tail"])].append(result["worker_id"])
                head_tail_map[(kg["head"], kg["tail"])].append(result["worker_id"])

        report["workers"].append(worker)
    
    add_stats_by_qtype(report, mcq_odd_results, "mcq_odd")
    add_stats_by_qtype(report, mcq_ds_results, "mcq_ds")
    add_stats_by_qtype(report, mcq_mt_results, "mcq_mt")

    for worker in tqdm(report["workers"], desc="Collecting worker stats"):
        worker["accuracy"] = round(np.mean([worker["questions"][qtype]["accuracy"] for qtype in ["mcq_odd", "mcq_ds", "mcq_mt"]]), 2)
        worker["f1"] = round(np.mean([worker["questions"][qtype]["f1"] for qtype in ["mcq_odd", "mcq_ds", "mcq_mt"]]), 2)
        worker["precision"] = round(np.mean([worker["questions"][qtype]["precision"] for qtype in ["mcq_odd", "mcq_ds", "mcq_mt"]]), 2)
        worker["recall"] = round(np.mean([worker["questions"][qtype]["recall"] for qtype in ["mcq_odd", "mcq_ds", "mcq_mt"]]), 2)

    dim_distribution = get_dim_distribution(knowledge)
    report["dim_distribution"] = dim_distribution

    report["head_rel_tail_map"] = [{f"{hrl[0]} | {hrl[1]} | {hrl[2]}": workers} for hrl, workers in head_rel_tail_map.items() if len(workers) > 1]
    report["head_tail_map"] = [{f"{ht[0]} || {ht[1]}": workers} for ht, workers in head_tail_map.items() if len(workers) > 1]

    results_path = pathlib.Path(args.results_path)
    report_path = results_path.parent / f"{results_path.stem}_report.json"

    write_json(report, report_path)

if __name__ == "__main__":
    main()