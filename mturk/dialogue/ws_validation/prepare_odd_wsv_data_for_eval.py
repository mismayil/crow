import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../../..")

from utils import read_json, write_json, clean_utterance, enrich_knowledge

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to aggregated results in json (results of WSV step)", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--vote-threshold", type=float, default=2, help="Threshold for votes.")

    args = parser.parse_args()
    results = read_json(args.results_path)

    eval_data = {}

    for result in tqdm(results, desc="Preparing results for evaluation"):
        eval_result = eval_data.get((result["data_id"], len(result["dialogue"])))

        if not eval_result:
            eval_result = {
                "data_id": result["data_id"],
                "instance_id": result["hit_id"],
                "dialogue": result["dialogue"],
                "final_turn_prefix": result["final_turn_prefix"],
                "final_turns": [{
                    "id": result["hit_id"],
                    "text": result["final_turn"],
                    "label": 1
                }]
            }
            eval_data[(result["data_id"], len(result["dialogue"]))] = eval_result

        if eval_result:
            for alt_target in result["alt_targets"]+result["new_alt_targets"]:
                votes = alt_target.get("votes")
                if votes is None or votes >= args.vote_threshold:
                    eval_result["final_turns"].append({
                        **alt_target,
                        "text": f"{result['final_turn_prefix']} {clean_utterance(alt_target['text'])}",
                        "label": 0,
                        "knowledge": enrich_knowledge(result["knowledge"])
                    })
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(list(eval_data.values()), eval_data_path)

if __name__ == "__main__":
    main()