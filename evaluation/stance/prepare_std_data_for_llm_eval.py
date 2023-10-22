import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import random

sys.path.append("../..")

from utils import read_json, write_json

INSTRUCTION_TEMPLATES  = {
    "bcq": "You are a helpful assistant for stance classification. Given a belief and an argument, answer whether the argument supports the belief. Answer only Yes or No.",
    "bcq_with_kg": "You are a helpful assistant for stance classification. Given a belief, an argument and a relevant knowledge, answer whether the argument supports the belief. Answer only Yes or No.",
    "bcq_cot": "You are a helpful assistant for stance classification. Given a belief and an argument, answer whether the argument supports the belief. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>.",
    "bcq_cot_with_kg": "You are a helpful assistant for stance classification. Given a belief, an argument and a relevant knowledge, answer whether the argument supports the belief. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>."
}

SHOT_TEMPLATES = {
    "bcq": "Example {index}:\n\nBelief:\n{belief}\n\nArgument:\n{argument}\n\nAnswer:{answer}",
    "bcq_with_kg": "Example {index}:\n\nBelief:\n{belief}\n\nArgument:\n{argument}\n\nKnowledge:\n{knowledge}\n\nAnswer:{answer}",
    "bcq_cot": "Example {index}:\n\nBelief:\n{belief}\n\nArgument:\n{argument}\n\nAnswer:\n{answer}",
    "bcq_cot_with_kg": "Example {index}:\n\nBelief:\n{belief}\n\nArgument:\n{argument}\n\nKnowledge:\n{knowledge}\n\nAnswer:\n{answer}"
}

SHOT_SAMPLES = [
    {
        "data_id": "expgraph-dev-68",
        "subdata_id": "304QEQWK0YAYUX4CH52DJ1ODGIMO0S",
        "belief": "cosmetic surgery should be banned.",
        "argument": "Cosmetic surgery is not worth the risk.",
        "stance": "support",
        "instances": [
            {
                "id": "35USIKEBN16DC7O98HNI2WTOSE96NB-0",
                "belief": "Cosmetic surgery should be allowed.",
                "argument": "Cosmetic surgery is not worth the risk.",
                "stance": "counter",
                "knowledge": [
                    {
                        "id": "expgraph-dev-68-ck-0",
                        "head": "cosmetic surgery",
                        "relation": "IsA",
                        "tail": "risky",
                        "dimension": "attribution",
                        "verbalized": "cosmetic surgery is a subtype or specific instance of risky"
                    }
                ],
                "perturbation": "belief",
                "answer": "Step 1: Analyze the belief\nAccording to the belief, cosmetic surgery should be allowed which might mean that it is not risky.\n\nStep 2: Analyze the argument\nThe argument states that cosmetic surgery is not worth the risk, so it assumes that there are risks involved, but it is not worth to do while taking the risk.\n\nStep 3: Compare the belief and argument\nThe belief supports cosmetic surgery, while the argument opposes it due to the risks involved.\n\nFinal Answer: <Answer>No</Answer>",
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "expgraph-dev-68",
        "subdata_id": "304QEQWK0YAYUX4CH52DJ1ODGIMO0S",
        "belief": "cosmetic surgery should be banned.",
        "argument": "Cosmetic surgery is not worth the risk.",
        "stance": "support",
        "instances": [
            {
                "id": "35USIKEBN16DC7O98HNI2WTOSE96NB-0",
                "belief": "Cosmetic surgery should be allowed.",
                "argument": "Cosmetic surgery is not worth the risk.",
                "stance": "counter",
                "knowledge": [
                    {
                        "id": "expgraph-dev-68-ck-0",
                        "head": "cosmetic surgery",
                        "relation": "IsA",
                        "tail": "risky",
                        "dimension": "attribution",
                        "verbalized": "cosmetic surgery is a subtype or specific instance of risky"
                    }
                ],
                "perturbation": "belief",
                "answer": "Step 1: Analyze the belief\nAccording to the belief, cosmetic surgery should be allowed which might mean that it is not risky.\n\nStep 2: Analyze the argument\nThe argument states that cosmetic surgery is not worth the risk, so it assumes that there are risks involved, but it is not worth to do while taking the risk.\n\nStep 3: Analyze the knowledge\nThe knowledge states that the cosmetic surgery is typically a risky thing to do.\n\nStep 3: Compare the belief and argument\nThe belief supports cosmetic surgery, while the argument opposes it due to the potential risks confirmed by the given knowledge.\n\nFinal Answer: <Answer>No</Answer>",
            }
        ],
        "type": "bcq_cot_with_kg"
    },
]

BINARY_CHOICE_TEMPLATES = ["bcq", "bcq_with_kg", "bcq_cot", "bcq_cot_with_kg"]

def prepare_sample_for_bcq(sample, shot_samples, template):
    shots = []
    shot_ids = []

    for idx, shot_sample in enumerate(shot_samples):
        instance = [i for i in shot_sample["instances"] if i.get("knowledge")][0]
        knowledge = instance.get("knowledge")
        answer = "Yes" if instance["stance"] == "support" else "No"

        if "answer" in instance:
            answer = instance["answer"]

        shot = SHOT_TEMPLATES[template].format(
            index=idx+1,
            belief=instance["belief"],
            argument=instance["argument"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=answer
        )
        shots.append(shot)
        shot_ids.append(instance["id"])

    shots_prompt = "\n\n".join(shots)

    eval_data = []

    for idx, instance in enumerate(sample["instances"]):
        # if instance["id"] in shot_ids:
        #     continue

        knowledge = instance.get("knowledge")
        final_shot = SHOT_TEMPLATES[template].format(
            index=len(shot_samples)+1,
            belief=instance["belief"],
            argument=instance["argument"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=""
        )
        prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
        eval_data.append({
            "data_id": sample["data_id"],
            "subdata_id": sample["subdata_id"],
            "instance_id": instance["id"],
            "belief": sample["belief"],
            "argument": sample["argument"],
            "stance": sample["stance"],
            "instance": instance,
            "prompt": prompt,
            "answer": ("Yes" if instance["stance"] == "support" else "No"),
            "type": template
        })

    return eval_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to eval data in json", required=True)
    parser.add_argument("--template", type=str, default="bcq")
    parser.add_argument("--num-shots", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")

    args = parser.parse_args()
    results = read_json(args.datapath)

    eval_data = []

    shot_samples = [sample for sample in SHOT_SAMPLES if sample["type"] == args.template][:args.num_shots]

    if not shot_samples:
        shot_samples = results[:args.num_shots]

    for result in tqdm(results, desc="Preparing results for LLM evaluation"):
        if args.template in BINARY_CHOICE_TEMPLATES:
            eval_data.extend(prepare_sample_for_bcq(result, shot_samples, args.template))
        else:
            raise ValueError(f"Template {args.template} not supported.")
    
    datapath = pathlib.Path(args.datapath)
    eval_data_path_stem = datapath.parent / f"{datapath.stem}_llm_{args.template}{args.suffix}"

    write_json(eval_data, eval_data_path_stem.with_suffix(".json"))
    pd.DataFrame(eval_data).to_csv(eval_data_path_stem.with_suffix(".csv"), index=False)

if __name__ == "__main__":
    main()