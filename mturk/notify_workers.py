import sys
import boto3
import argparse
from tqdm import tqdm

sys.path.append("..")

from utils import read_json, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-path", type=str, help="Path to worker details json file.", required=True)
    parser.add_argument("--subject-attr", type=str, help="Subject attribute.", default="feedback_subject")
    parser.add_argument("--feedback-attr", type=str, help="Feedback attribute.", default="requester_feedback")
    parser.add_argument("--feedback", type=str, help="Feedback to send to workers.", default=None)
    parser.add_argument("--subject", type=str, help="Feedback subject.", default="Feedback")

    args = parser.parse_args()

    session = boto3.Session(profile_name='mturk')
    mturk_client = session.client('mturk')

    workers = read_json(args.workers_path)
    num_notifications = 0

    for worker in tqdm(workers, desc="Notifying workers"):
        subject = worker.get(args.subject_attr, args.subject)
        feedback = worker.get(args.feedback_attr, args.feedback)
        notified = worker.get("notified", False)

        if feedback and not notified:
            num_notifications += 1

            response = mturk_client.notify_workers(
                Subject=subject,
                MessageText=feedback,
                WorkerIds=[
                    worker["worker_id"]
                ]
            )

            if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                print("Failed to notify worker: {}".format(worker["worker_id"]))
            else:
                worker["notified"] = True
    
    write_json(workers, args.workers_path)

    print("Notified {} workers.".format(num_notifications))

if __name__ == "__main__":
    main()