import sys
import boto3
import argparse
from tqdm import tqdm

sys.path.append("../..")

from utils import read_json

CKA_POOL_ID = "3RDXJZR9AB8M5RP2QLLDS39FR1JYCZ"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-path", type=str, help="Path to worker details json file.", required=True)
    parser.add_argument("--pool-id", type=str, help="Pool ID to assign qualification to.", default=CKA_POOL_ID)
    parser.add_argument("--send-notification", action="store_true", help="Send notification to workers.", default=False)
    parser.add_argument("--qualified-only", action="store_true", help="Only assign qualification to qualified workers.", default=False)

    args = parser.parse_args()

    session = boto3.Session(profile_name='mturk')
    mturk_client = session.client('mturk')

    workers = read_json(args.workers_path)
    num_qualified = 0

    for worker in tqdm(workers, desc="Assigning qualification to workers"):
        if not args.qualified_only or worker["qualified"]:
            num_qualified += 1
            response = mturk_client.associate_qualification_with_worker(
                QualificationTypeId=args.pool_id,
                WorkerId=worker["worker_id"],
                IntegerValue=worker["score"],
                SendNotification=args.send_notification
            )

            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                print("Failed to assign qualification to worker: {}".format(worker["worker_id"]))
    
    print("Assigned qualification to {} workers.".format(num_qualified))

if __name__ == "__main__":
    main()