import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to csv results.", required=True)

    args = parser.parse_args()
    results = read_json(args.results_path)
    agg_results = {}

    for result in tqdm(results, desc="Aggregating results", total=len(results)):
        agg_result = agg_results.get(result["hit_id"])

        if not agg_result:
            agg_result = {
                "data_id": result["data_id"],
                "hit_id": result["hit_id"],
                "assignments": [result["assignment_id"]],
                "workers": [result["worker_id"]],
                "dialogue": result["dialogue"],
                "knowledge": [{"id": kg["id"], "head": kg["head"], "relation": kg["relation"], "tail": kg["tail"], "head_turn": kg.get("head_turn"), "tail_turn": kg.get("tail_turn"), "votes": int(kg.get("selected", False))} for kg in result["knowledge"]],
                "new_knowledge": result.get("new_knowledge", []),
                "option_none": int(result.get("option_none", False))
            }
            agg_results[result["hit_id"]] = agg_result
        else:
            agg_result["assignments"].append(result["assignment_id"])
            agg_result["workers"].append(result["worker_id"])

            for k_index, kg in enumerate(result["knowledge"]):
                agg_result["knowledge"][k_index]["votes"] += int(kg.get("selected", False))
            
            agg_result["option_none"] += int(result.get("option_none", False))
            agg_result["new_knowledge"] += result.get("new_knowledge", [])
    
    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"

    write_json(list(agg_results.values()), agg_results_path)

if __name__ == "__main__":
    main()