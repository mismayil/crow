import sys
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
import random

sys.path.append("../..")

from utils import read_json, clean, parse_turn

MTURK_DIALOGUE_TEMPLATE = "<strong>{index}. {speaker}:</strong> {utterance}<br>"

def process_dialogue(dialogue):
    utterances = []

    for index, turn in enumerate(dialogue):
        speaker, utterance = parse_turn(turn)
        utterances.append(MTURK_DIALOGUE_TEMPLATE.format(index=index+1, speaker="#Person1#" if index % 2 == 0 else "#Person2#", utterance=clean(utterance)))
    
    return "\n".join(utterances)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to data.", required=True)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")

    args = parser.parse_args()
    data = read_json(args.datapath)
    mturk_data = []

    for s_index, sample in tqdm(enumerate(data), total=len(data)):
        mturk_sample = {"id": f"cka_qual_{s_index}"}

        for q_index, question in enumerate(sample["mcq_odd_questions"]):
            mturk_sample[f"mcq_odd_q{q_index+1}_dialogue"] = process_dialogue(question["dialogue"])

            for o_index, option in enumerate(question["options"]):
                mturk_sample[f"mcq_odd_q{q_index+1}_opt{o_index+1}_head"] = option["head"]
                mturk_sample[f"mcq_odd_q{q_index+1}_opt{o_index+1}_rel"] = option["relation"]
                mturk_sample[f"mcq_odd_q{q_index+1}_opt{o_index+1}_tail"] = option["tail"]
                mturk_sample[f"mcq_odd_q{q_index+1}_opt{o_index+1}_label"] = option["label"]

        for q_index, question in enumerate(sample["mcq_ds_questions"]):
            mturk_sample[f"mcq_ds_q{q_index+1}_dialogue"] = process_dialogue(question["dialogue"])
            mturk_sample[f"mcq_ds_q{q_index+1}_summary"] = question["summary"]
            
            for o_index, option in enumerate(question["options"]):
                mturk_sample[f"mcq_ds_q{q_index+1}_opt{o_index+1}_head"] = option["head"]
                mturk_sample[f"mcq_ds_q{q_index+1}_opt{o_index+1}_rel"] = option["relation"]
                mturk_sample[f"mcq_ds_q{q_index+1}_opt{o_index+1}_tail"] = option["tail"]
                mturk_sample[f"mcq_ds_q{q_index+1}_opt{o_index+1}_label"] = option["label"]
        
        for q_index, question in enumerate(sample["mcq_mt_questions"]):
            coin = random.choice([0, 1])
            sent1 = question["plausible_sentence"] if coin == 0 else question["implausible_sentence"]
            sent2 = question["implausible_sentence"] if coin == 0 else question["plausible_sentence"]
            mturk_sample[f"mcq_mt_q{q_index+1}_sentence1"] = sent1
            mturk_sample[f"mcq_mt_q{q_index+1}_sentence2"] = sent2
            mturk_sample[f"mcq_mt_q{q_index+1}_implausible_label"] = 2-coin
            
            for o_index, option in enumerate(question["options"]):
                mturk_sample[f"mcq_mt_q{q_index+1}_opt{o_index+1}_head"] = option["head"]
                mturk_sample[f"mcq_mt_q{q_index+1}_opt{o_index+1}_rel"] = option["relation"]
                mturk_sample[f"mcq_mt_q{q_index+1}_opt{o_index+1}_tail"] = option["tail"]
                mturk_sample[f"mcq_mt_q{q_index+1}_opt{o_index+1}_label"] = option["label"]

        for q_index, question in enumerate(sample["oq_odd_questions"]):
            mturk_sample[f"oq_odd_q{q_index+1}_dialogue"] = process_dialogue(question["dialogue"])
            mturk_sample[f"oq_odd_q{q_index+1}_num_turns"] = len(question["dialogue"])

        mturk_data.append(mturk_sample)
    
    pathlib.Path("data").mkdir(exist_ok=True, parents=True)
    datapath = pathlib.Path(args.datapath)
    pd.DataFrame(mturk_data).to_csv(f"data/mturk_data_{datapath.stem}{args.suffix}.csv", index=False)

if __name__ == "__main__":
    main()