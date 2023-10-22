import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, prepare_dialogue_for_eval, enrich_knowledge

def add_knowledge_to_positive_summaries(eval_result, knowledge):
    positive_summs = [summary for summary in eval_result["summaries"] if summary["label"] == 1]

    for positive_summ in positive_summs:
        for kg in knowledge:
            kg_exists = [k for k in positive_summ["knowledge"] if k["id"] == kg["id"]]

            if not kg_exists:
                positive_summ["knowledge"].append(kg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to aggregated results in json (results of WSV step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--vote-threshold", type=float, default=2, help="Threshold for votes.")

    args = parser.parse_args()
    results = read_json(args.results_path)

    eval_data = {}

    for result in tqdm(results, desc="Preparing results for evaluation"):
        eval_result = eval_data.get(result["data_id"])

        if not eval_result:
            eval_result = {
                "data_id": result["data_id"],
                "subdata_id": result["hit_id"],
                "dialogue": prepare_dialogue_for_eval(result["dialogue"]),
                "summaries": [{
                    "id": result["hit_id"],
                    "text": result["summary"],
                    "label": 1,
                    "knowledge": []
                }]
            }
            eval_data[result["data_id"]] = eval_result

        if eval_result:
            for alt_target in result["alt_targets"]+result["new_alt_targets"]:
                votes = alt_target.get("votes")
                if votes is None or votes >= args.vote_threshold:
                    enriched_knowledge = enrich_knowledge(result["knowledge"])
                    eval_result["summaries"].append({
                        **alt_target,
                        "label": 0,
                        "knowledge": enriched_knowledge
                    })
                    add_knowledge_to_positive_summaries(eval_result, enriched_knowledge)
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()