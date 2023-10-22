import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, enrich_knowledge

def add_knowledge_to_other_instances(eval_result, knowledge, instance_id):
    other_instances = [instance for instance in eval_result["instances"] if instance["id"] == instance_id]

    for other_instance in other_instances:
        for kg in knowledge:
            kg_exists = [k for k in other_instance["knowledge"] if k["id"] == kg["id"]]

            if not kg_exists:
                other_instance["knowledge"].append(kg)

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
                "belief": result["belief"],
                "argument": result["argument"],
                "stance": result["stance"],
                "instances": [{
                    "id": result["data_id"],
                    "belief": result["belief"],
                    "argument": result["argument"],
                    "stance": result["stance"],
                    "knowledge": []
                }]
            }
            eval_data[result["data_id"]] = eval_result

        if eval_result:
            for alt_target in result["alt_targets"]+result.get("new_alt_targets", []):
                votes = alt_target.get("votes")

                if votes is None or votes >= args.vote_threshold:
                    if "knowledge_ids" in alt_target:
                        selected_knowledge = [kg for kg in result["knowledge"] if kg["id"] in alt_target["knowledge_ids"]]
                    else:
                        selected_knowledge = result["knowledge"]
                    perturbation = alt_target["perturbation"]
                    belief = alt_target["text"] if perturbation == "belief" else result["belief"]
                    argument = alt_target["text"] if perturbation == "argument" else result["argument"]
                    enriched_knowledge = enrich_knowledge(selected_knowledge)
                    eval_result["instances"].append({
                        "id": alt_target["id"],
                        "belief": belief,
                        "argument": argument, 
                        "stance": "counter" if result["stance"] == "support" else "support",
                        "knowledge": enriched_knowledge,
                        "perturbation": perturbation,
                        "votes": votes
                    })
                    add_knowledge_to_other_instances(eval_result, enriched_knowledge, result["data_id"])
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()