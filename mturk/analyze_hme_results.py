import sys
import argparse
from tqdm import tqdm
import numpy as np
import pathlib
from sklearn.metrics import f1_score
import pandas as pd

sys.path.append("..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, help="Path to aggregated HME json results.", required=True)
    parser.add_argument("--resolution-path", type=str, help="Path to resolved HME csv results.")
    parser.add_argument("--num-annotators", type=int, help="Number of annotators per data point.", default=2)
    
    args = parser.parse_args()

    results = read_json(args.results_path)
    resolutions = pd.read_csv(args.resolution_path)

    resolutions = resolutions.to_dict("records")

    report = {
        "num_results": len(results),
    }

    references = []
    predictions = []
    per_annotator_references = [[] for _ in range(args.num_annotators)]
    per_annotator_predictions = [[] for _ in range(args.num_annotators)]

    per_data_predictions = {}
    per_data_annotator_predictions = [{} for _ in range(args.num_annotators)]

    resolution_map = {}

    if resolutions:
        for resolution in resolutions:
            resolution_map[resolution["hit_id"]] = resolution
    
    for result in tqdm(results, total=len(results), desc="Analyzing results"):
        references.append(result["label"])
        
        for i, _ in enumerate(result["answers"]):
            per_annotator_references[i].append(result["label"])
        
        if result["agreement"] and len(result["answers"]) == args.num_annotators:
            final_answer = result["answers"][0]
        else:
            if resolutions:
                resolution = resolution_map[result["hit_id"]]
                final_answer = resolution["prediction"]

        predictions.append(final_answer)

        for i, answer in enumerate(result["answers"]):
            per_annotator_predictions[i].append(answer)

        per_data_predictions[result["data_id"]] = per_data_predictions.get(result["data_id"], True) & (final_answer == result["label"])

        for i, answer in enumerate(result["answers"]):
            per_data_annotator_predictions[i][result["data_id"]] = per_data_annotator_predictions[i].get(result["data_id"], True) & (answer == result["label"])
    
    report["f1_score"] = f1_score(references, predictions, average="macro")
    report["exact_match"] = np.mean(list(per_data_predictions.values()))

    annotator_f1_scores = []
    annotator_exact_match_scores = []

    for ann_refs, ann_preds, data_ann_preds in zip(per_annotator_references, per_annotator_predictions, per_data_annotator_predictions):
        annotator_f1_scores.append(f1_score(ann_refs, ann_preds, average="macro"))
        annotator_exact_match_scores.append(np.mean(list(data_ann_preds.values())))

    report["annotator_f1_scores"] = annotator_f1_scores
    report["annotator_exact_match_scores"] = annotator_exact_match_scores
    report["avg_annotator_f1_score"] = np.mean(annotator_f1_scores)
    report["avg_annotator_exact_match"] = np.mean(annotator_exact_match_scores)

    results_path = pathlib.Path(args.results_path)
    report_path = results_path.parent / f"{results_path.stem}_report.json"

    write_json(report, report_path)

if __name__ == "__main__":
    main()