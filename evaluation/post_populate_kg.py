import argparse
import sys
import pathlib
from collections import defaultdict
from tqdm import tqdm


sys.path.append("../")

from utils import read_json, write_json, find_json_files

TASK_TARGET_MAP = {
    "dialogue": "final_turn", 
    "summarization": "summary", 
    "stance": "instance" , 
    "intent": "intent", 
    "safety": "action", 
    "translation": "translation",
    "mt_en_fr": "translation", 
    "mt_en_de": "translation", 
    "mt_en_ru": "translation", 
    "mt_zh_en": "translation",
}

def populate_kg(results, task):
    data = results["data"]
    data_kg_map = defaultdict(list)
    target_attr = TASK_TARGET_MAP[task]

    for d in tqdm(data, total=len(data), desc="Hashing KGs", leave=False):
        target = d[target_attr]

        if "knowledge" in target:
            for kg in target["knowledge"]:
                kg_exists = [k for k in data_kg_map[d["subdata_id"]] if k["id"] == kg["id"]]
                if not kg_exists:
                    data_kg_map[d["subdata_id"]].append(kg)
    
    for d in tqdm(data, total=len(data), desc="Populating KGs", leave=False):
        target = d[target_attr]
        if d["subdata_id"] in data_kg_map and "knowledge" not in target:
            target["knowledge"] = data_kg_map[d["subdata_id"]]
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, help="Path to evaluation results dir", required=True)
    parser.add_argument("--task", type=str, help="Name of the task", default=None)

    args = parser.parse_args()

    output_files = find_json_files(args.results_dir)

    for output_file in tqdm(output_files, total=len(output_files), desc="Processing files", leave=False):
        print("Processing", output_file)
        output_path = pathlib.Path(output_file)

        results = read_json(output_file)

        if "data" in results:
            task = [task for task in TASK_TARGET_MAP if task in output_file][0]

            if not args.task or task == args.task:
                with_kg = populate_kg(results, task)
                write_json(with_kg, output_path)

if __name__ == "__main__":
    main()