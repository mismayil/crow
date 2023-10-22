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
                "dialogue": result["dialogue"],
                "summary": result["summary"],
                "knowledge": result["knowledge"],
                "alt_targets": [{"id": summary["id"], "text": summary["text"], "votes": int(summary.get("selected", False))} for summary in result["alt_targets"]],
                "new_alt_targets": result.get("new_alt_targets", []),
                "option_none": int(result.get("option_none", False))
            }
            agg_results[result["hit_id"]] = agg_result
        else:
            for t_index, turn in enumerate(result["alt_targets"]):
                agg_result["alt_targets"][t_index]["votes"] += int(turn.get("selected", False))
            
            agg_result["option_none"] += int(result.get("option_none", False))
            agg_result["new_alt_targets"] += result.get("new_alt_targets", [])
    
    results_path = pathlib.Path(args.results_path)
    agg_results_path = results_path.parent / f"{results_path.stem}_agg.json"

    write_json(list(agg_results.values()), agg_results_path)

if __name__ == "__main__":
    main()