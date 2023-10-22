import sys
import argparse
import pathlib
from tqdm import tqdm

sys.path.append("../..")

from utils import read_json, write_json, enrich_kg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to aggregated results in json (results of CKV step)", required=True)
    parser.add_argument("--source-path", type=str, help="Path to source data", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--vote-threshold", type=float, default=2, help="Threshold for votes.")

    args = parser.parse_args()
    results = read_json(args.results_path)
    source_data = read_json(args.source_path)

    eval_data = []
    source_map = {}

    for data in source_data:
        if "qID" in data:
            source_map[data["qID"]] = data
        else:
            source_map[data["id"]] = data
    
    for result in tqdm(results, desc="Preparing results for evaluation"):
        source = source_map[result["data_id"]]
        eval_result = {
            "data_id": result["data_id"],
            "source": None,
            "targets": [],
            "knowledge": []
        }
        
        if "plausible_sentence" in result:
            eval_result["source"] = source["chinese_source"]
            eval_result["targets"].append({
                "text": result["plausible_sentence"],
                "label": 1
            })
            eval_result["targets"].append({
                "text": result["implausible_sentence"],
                "label": 0
            })
        else:
            eval_result["source"] = result["sentence"]
            eval_result["targets"].append({
                "text": source[f"translation{source['answer']}"],
                "label": 1
            })
            eval_result["targets"].append({
                "text": source[f"translation{3-source['answer']}"],
                "label": 0
            })
        
        for kg in result["knowledge"]+result.get("new_knowledge", []):
            votes = kg.get("votes")
            if votes is None or votes >= args.vote_threshold:
                eval_result["knowledge"].append(enrich_kg(kg))
        
        if eval_result["knowledge"]:
            eval_data.append(eval_result)
    
    results_path = pathlib.Path(args.results_path)
    eval_data_path = results_path.parent / f"{results_path.stem}_eval{args.suffix}.json"

    write_json(eval_data, eval_data_path)

if __name__ == "__main__":
    main()