import sys
import argparse
from tqdm import tqdm
import pathlib

sys.path.append("..")

from utils import read_json, write_json

def add_workers_to_map(workers, worker_map, prefix=""):
    for worker in tqdm(workers, total=len(workers), desc=f"Analyzing {prefix} workers"):
        c_worker = worker_map.get(worker["worker_id"])

        if not c_worker:
            c_worker = {"worker_id": worker["worker_id"]}
            worker_map[worker["worker_id"]] = c_worker
        
        c_worker[prefix] = worker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cka-report-path", type=str, help="Path to CKA report.")
    parser.add_argument("--ckv-report-path", type=str, help="Path to CKV report.")
    parser.add_argument("--wsg-report-path", type=str, help="Path to WSG report.")
    parser.add_argument("--wsv-report-path", type=str, help="Path to WSV report.")
    parser.add_argument("--output-dir", type=str, help="Path to output directory.", default="worker_reports")
    parser.add_argument("--suffix", type=str, help="Suffix for report.", default="")

    args = parser.parse_args()

    worker_map = {}

    if args.cka_report_path:
        cka_report = read_json(args.cka_report_path)
        add_workers_to_map(cka_report["workers"], worker_map, prefix="cka")

    if args.ckv_report_path:
        ckv_report = read_json(args.ckv_report_path)
        add_workers_to_map(ckv_report["workers"], worker_map, prefix="ckv")
    
    if args.wsg_report_path:
        wsg_report = read_json(args.wsg_report_path)
        add_workers_to_map(wsg_report["workers"], worker_map, prefix="wsg")
    
    if args.wsv_report_path:
        wsv_report = read_json(args.wsv_report_path)
        add_workers_to_map(wsv_report["workers"], worker_map, prefix="wsv")

    pathlib.Path(args.output_dir).mkdir(exist_ok=True, parents=True)

    write_json(worker_map, f"{args.output_dir}/workers_report{args.suffix}.json")

if __name__ == "__main__":
    main()