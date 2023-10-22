import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import random

sys.path.append("../..")

from utils import read_json, write_json

INSTRUCTION_TEMPLATES  = {
    "bcq": "You are a helpful assistant for intent classification. Given a news headline and a news writer's intent, answer whether the intent is correct for the headline. Answer only Yes or No.",
    "bcq_with_kg": "You are a helpful assistant for intent classification. Given a news headline, a news writer's intent and a relevant knowledge, answer whether the intent is correct for the headline. Answer only Yes or No.",
    "mcq": "You are a helpful assistant for intent classification. Given a news headline and multiple news writer intents, select ones which are correct for the headline. Select all that apply.",
    "bcq_cot": "You are a helpful assistant for intent classification. Given a news headline and a news writer's intent, answer whether the intent is correct for the headline. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>.",
    "bcq_cot_with_kg": "You are a helpful assistant for intent classification. Given a news headline, a news writer's intent and a relevant knowledge, answer whether the intent is correct for the headline. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>."
}

SHOT_TEMPLATES = {
    "bcq": "Example {index}:\n\nHeadline:\n{headline}\n\nIntent:\n{intent}\n\nAnswer:{answer}",
    "bcq_with_kg": "Example {index}:\n\nHeadline:\n{headline}\n\nIntent:\n{intent}\n\nKnowledge:\n{knowledge}\n\nAnswer:{answer}",
    "mcq": "Example {index}:\n\nHeadline:\n{headline}\n\nIntents:\n{intents}\n\nAnswer:{answer}",
    "bcq_cot": "Example {index}:\n\nHeadline:\n{headline}\n\nIntent:\n{intent}\n\nAnswer:\n{answer}",
    "bcq_cot_with_kg": "Example {index}:\n\nHeadline:\n{headline}\n\nIntent:\n{intent}\n\nKnowledge:\n{knowledge}\n\nAnswer:\n{answer}",
}

SHOT_SAMPLES = [
    {
        "data_id": "mrf-covid-test-1129",
        "subdata_id": "31SIZS5W6I5PF31RQVDWS2K0475RQZ",
        "headline": "Authorities will delay vaccines in Andalusia. They bought millions of syringes that will not work to distribute the COVID-19 vaccine",
        "intents": [
            {
                "id": "mrf-covid-test-1129-intent-0",
                "text": "the vaccine requires specific needles to apply",
                "label": 1,
                "knowledge": [
                    {
                        "id": "3YDTZAI2W76WJDD6K6W58RW7TE014Z-1",
                        "head": "millions of syringes that will not work",
                        "relation": "Implies",
                        "tail": "requires specific needles",
                        "head_from": "headline",
                        "tail_from": "intent",
                        "dimension": "causal",
                        "verbalized": "millions of syringes that will not work implies requires specific needles"
                    }
                ],
                "answer": "Step 1: Analyze the headline\nThe headline states that authorities in Andalusia will delay vaccines because they bought millions of syringes that will not work to distribute the COVID-19 vaccine. This shows that there is a incompatibility between the bought syringes and syringes required for the vaccine.\n\nStep 2: Analyze the intent\nThe intent states that the vaccine requires specific needles to apply. This means standard syringes might not be suitable.\n\nStep 3: Compare the headline and intent\nThe headline implies that the syringes purchased are not suitable for distributing the COVID-19 vaccine, which aligns with the intent stating that specific needles are required to apply the vaccine.\n\nSo, the given intent is the correct one for this headline.\n\nFinal Answer: <Answer>Yes</Answer>"
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "mrf-covid-test-1129",
        "subdata_id": "31SIZS5W6I5PF31RQVDWS2K0475RQZ",
        "headline": "Authorities will delay vaccines in Andalusia. They bought millions of syringes that will not work to distribute the COVID-19 vaccine",
        "intents": [
            {
                "id": "3JC6VJ2SAL9A9KU6UU88OKVOYIUA5N-0",
                "text": "the vaccine requires standard needles to apply",
                "votes": 3,
                "label": 0,
                "knowledge": [
                    {
                        "id": "3YDTZAI2W76WJDD6K6W58RW7TE014Z-1",
                        "head": "millions of syringes that will not work",
                        "relation": "Implies",
                        "tail": "requires specific needles",
                        "head_from": "headline",
                        "tail_from": "intent",
                        "dimension": "causal",
                        "verbalized": "millions of syringes that will not work implies requires specific needles"
                    }
                ],
                "answer": "Step 1: Analyze the headline\n- Authorities will delay vaccines in Andalusia\n- They bought millions of syringes that will not work to distribute the COVID-19 vaccine\n\nStep 2: Analyze the intent\n- The vaccine requires standard needles to apply which means any bought needle can be used to distribute COVID-19 vaccine\n\nStep 3: Analyze the knowledge\n- Millions of syringes that will not work imply that the vaccine might require specific needles\n\nStep 4: Determine if the intent is correct for the headline\nAccording to the headline, there is an incompatibility between the bought syringes and the syringes required for the vaccine so they won't work and delay the process. Given knowledge tells us that this incompatibility might imply that the vaccine requires specific needles. So, this is the true intent behind this headline while the given intent assumes the vaccine requires standard needles.\n\nSo the intent is incorrect for the headline.\n\nFinal Answer: <Answer>No</Answer>"
            }
        ],
        "type": "bcq_cot_with_kg"
    }
]
    
BINARY_CHOICE_TEMPLATES = ["bcq", "bcq_with_kg", "bcq_cot", "bcq_cot_with_kg"]
MULTIPLE_CHOICE_TEMPLATES = ["mcq"]

def prepare_sample_for_bcq(sample, shot_samples, template):
    shots = []
    shot_ids = []

    for idx, shot_sample in enumerate(shot_samples):
        intent = [i for i in shot_sample["intents"] if i.get("knowledge")][0]
        knowledge = intent.get("knowledge")
        answer = "Yes" if intent["label"] == 1 else "No"

        if "answer" in intent:
            answer = intent["answer"]

        shot = SHOT_TEMPLATES[template].format(
            index=idx+1,
            headline=shot_sample["headline"],
            intent=intent["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=answer
        )
        shots.append(shot)
        shot_ids.append(intent["id"])

    shots_prompt = "\n\n".join(shots)

    eval_data = []

    for idx, intent in enumerate(sample["intents"]):
        # if intent["id"] in shot_ids:
        #     continue

        knowledge = intent.get("knowledge")
        final_shot = SHOT_TEMPLATES[template].format(
            index=len(shot_samples)+1,
            headline=sample["headline"],
            intent=intent["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=""
        )
        prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
        eval_data.append({
            "data_id": sample["data_id"],
            "subdata_id": sample["subdata_id"],
            "instance_id": intent["id"],
            "headline": sample["headline"],
            "intent": intent,
            "prompt": prompt,
            "answer": ("Yes" if intent["label"] == 1 else "No"),
            "type": template
        })

    return eval_data

def prepare_sample_for_mcq(sample, shot_samples, template):
    eval_data = []
    shots = []

    for s_idx, shot_sample in enumerate(shot_samples):
        intents = random.sample(shot_sample["intents"], len(shot_sample["intents"]))

        shot = SHOT_TEMPLATES[template].format(
            index=s_idx+1,
            headline=shot_sample["headline"],
            intents="\n".join([f"{i_idx+1}. {intent['text']}" for i_idx, intent in enumerate(intents)]),
            answer=",".join([str(i_idx+1) for i_idx, intent in enumerate(intents) if intent["label"] == 1])
        )
        shots.append(shot)

    shots_prompt = "\n\n".join(shots)

    intents = random.sample(sample["intents"], len(sample["intents"]))

    final_shot = SHOT_TEMPLATES[template].format(
        index=len(shot_samples)+1,
        headline=sample["headline"],
        intents="\n".join([f"{i_idx+1}. {intent['text']}" for i_idx, intent in enumerate(intents)]),
        answer=""
    )

    prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
    eval_data.append({
        "data_id": sample["data_id"],
        "subdata_id": sample["subdata_id"],
        "instance_id": sample["subdata_id"],
        "headline": sample["headline"],
        "intent": intents,
        "prompt": prompt,
        "answer": ",".join([str(i_idx+1) for i_idx, intent in enumerate(intents) if intent["label"] == 1]),
        "type": template,
        "num_options": len(intents)
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
        elif args.template in MULTIPLE_CHOICE_TEMPLATES:
            eval_data.extend(prepare_sample_for_mcq(result, shot_samples, args.template))
        else:
            raise ValueError(f"Invalid template: {args.template}")

    
    datapath = pathlib.Path(args.datapath)
    eval_data_path_stem = datapath.parent / f"{datapath.stem}_llm_{args.template}{args.suffix}"

    write_json(eval_data, eval_data_path_stem.with_suffix(".json"))
    pd.DataFrame(eval_data).to_csv(eval_data_path_stem.with_suffix(".csv"), index=False)

if __name__ == "__main__":
    main()