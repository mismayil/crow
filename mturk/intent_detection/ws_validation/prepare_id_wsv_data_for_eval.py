import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, enrich_knowledge

def add_knowledge_to_positive_intents(eval_result, knowledge, intent_id):
    positive_intents = [intent for intent in eval_result["intents"] if intent["id"] == intent_id]

    for positive_intent in positive_intents:
        for kg in knowledge:
            kg_exists = [k for k in positive_intent["knowledge"] if k["id"] == kg["id"]]

            if not kg_exists:
                positive_intent["knowledge"].append(kg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to results in json (results of WSG step)", required=True)
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
                "headline": result["headline"],
                "intents": []
            }
            eval_data[result["data_id"]] = eval_result

        if eval_result:
            true_intent_exists = any([intent["id"] == result["intent_id"] for intent in eval_result["intents"]])

            if not true_intent_exists:
                eval_result["intents"].append({
                    "id": result["intent_id"],
                    "text": result["intent"],
                    "label": 1,
                    "knowledge": []
                })

            for alt_target in result["alt_targets"]+result.get("new_alt_targets", []):
                votes = alt_target.get("votes")

                if votes is None or votes >= args.vote_threshold:
                    if "knowledge_ids" in alt_target:
                        selected_knowledge = [kg for kg in result["knowledge"] if kg["id"] in alt_target["knowledge_ids"]]
                    else:
                        selected_knowledge = result["knowledge"]
                    enriched_knowledge = enrich_knowledge(selected_knowledge)
                    eval_result["intents"].append({
                        **alt_target,
                        "label": 0,
                        "knowledge": enriched_knowledge
                    })
                    add_knowledge_to_positive_intents(eval_result, enriched_knowledge, result["intent_id"])
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()