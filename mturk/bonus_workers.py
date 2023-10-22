import sys
import boto3
import argparse
from tqdm import tqdm

sys.path.append("..")

from utils import read_json, write_json

BONUS_AMOUNT = 0.1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-path", type=str, help="Path to worker details json file.", required=True)
    parser.add_argument("--report-path", type=str, help="Path to report json file.")
    parser.add_argument("--bonus-amount", type=str, help="Bonus amount to give.", default=BONUS_AMOUNT)
    parser.add_argument("--bonus-reason", type=str, help="Reason for giving bonus.", default="Bonus for additional work.")
    parser.add_argument("--reason-attr", type=str, help="Bonus reason attribute", default="bonus_reason")
    parser.add_argument("--qualified-only", action="store_true", help="Give bonus only to qualified workers.", default=False)
    parser.add_argument("--quality-threshold", type=float, help="Quality threshold to check.", default=0)

    args = parser.parse_args()

    session = boto3.Session(profile_name='mturk')
    mturk_client = session.client('mturk')

    workers = read_json(args.workers_path)
    report_map = {}

    if args.report_path:
        report = read_json(args.report_path)
        report_map = {worker["worker_id"]: worker for worker in report["workers"]}

    num_bonuses = 0

    for worker in tqdm(workers, desc="Sending bonus to workers"):
        bonus_sent = worker.get("bonus_sent", False)
        if not bonus_sent and worker["bonus"] > 0 and (not args.qualified_only or worker.get("qualified", False)):
            worker_report = report_map.get(worker["worker_id"])

            if worker_report and worker_report["quality"] < args.quality_threshold:
                print("Skipping worker {}. Quality below threshold.".format(worker["worker_id"]))
                continue

            num_bonuses += 1
            reason = worker.get(args.reason_attr, args.bonus_reason)
            bonus = round(worker["bonus"]*args.bonus_amount, 2)

            assert bonus > 0 and bonus < 5, "Bonus amount must be between 0 and 5"

            print("Sending bonus of ${} to worker {} for reason: {}".format(bonus, worker["worker_id"], reason))
            
            try:
                response = mturk_client.send_bonus(
                    WorkerId=worker["worker_id"],
                    BonusAmount=str(bonus),
                    AssignmentId=worker["assignment_id"],
                    Reason=(reason if reason else args.bonus_reason),
                    UniqueRequestToken=worker["assignment_id"]
                )
            except Exception as e:
                if worker["assignment_id"] in str(e):
                    print("Skipping worker {}. Already received bonus for this assignment.".format(worker["worker_id"]))
                    continue

            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                print("Failed to bonus worker: {}".format(worker["worker_id"]))
            else:
                worker["bonus_sent"] = True

    write_json(workers, args.workers_path)

    print("Sent bonus for {} annotations.".format(num_bonuses))

if __name__ == "__main__":
    main()