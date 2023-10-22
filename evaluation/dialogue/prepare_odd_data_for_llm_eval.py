import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import random

sys.path.append("../..")

from utils import read_json, write_json, prepare_dialogue_for_eval, prepare_dialogue_turn_for_eval

INSTRUCTION_TEMPLATES  = {
    "bcq": "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B, answer whether the given response can plausibly follow this dialogue. Answer only 'Yes' or 'No'.",
    "bcq_with_kg": "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B and a relevant knowledge about this dialogue, answer whether the given response can plausibly follow this dialogue. Answer only 'Yes' or 'No'.",
    "mcq": "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B, select responses that can plausibly follow the dialogue. Select all that apply.",
    "bcq_cot": "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B, answer whether the given response can plausibly follow this dialogue. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>.",
    "bcq_cot_with_kg": "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B and a relevant knowledge about this dialogue, answer whether the given response can plausibly follow this dialogue. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>."
}

SHOT_TEMPLATES = {
    "bcq": "Example {index}:\n\nDialogue:\n{dialogue}\n\nResponse:\n{response}\n\nAnswer:{answer}",
    "bcq_with_kg": "Example {index}:\n\nDialogue:\n{dialogue}\n\nKnowledge:\n{knowledge}\n\nResponse:\n{response}\n\nAnswer:{answer}",
    "mcq": "Example {index}:\n\nDialogue:\n{dialogue}\n\nResponses:\n{responses}\n\nAnswer:{answer}",
    "bcq_cot": "Example {index}:\n\nDialogue:\n{dialogue}\n\nResponse:\n{response}\n\nAnswer:\n{answer}",
    "bcq_cot_with_kg": "Example {index}:\n\nDialogue:\n{dialogue}\n\nKnowledge:\n{knowledge}\n\nResponse:\n{response}\n\nAnswer:\n{answer}"
}

SHOT_SAMPLES = [
    {
        "data_id": "daily-dialogue-0001",
        "dialogue": [
            "A: ( Before Christmas Party ) Are you ready for the Christmas party tonight",
            "B: Almost. I have to get dressed. It's a formal party and I have special party make up!",
            "A: Use this lipstick and it will make your lips shine!"
        ],
        "final_turns": [
            {
                "id": "3X4JMASXCWZGXNDX322S20J3WU40BO-0",
                "knowledge": [
                    {
                        "dimension": "temporal",
                        "head": "gift exchange",
                        "head_turn": 4,
                        "id": "daily-dialogue-0001-5",
                        "relation": "HappensIn",
                        "tail": "Christmas party",
                        "tail_turn": 1,
                        "verbalized": "gift exchange happens in Christmas party"
                    }
                ],
                "label": 0,
                "text": "B: Great! Uh, remember that there's a rocket launch, too. We all have to bring a gift.",
                "votes": 1,
                "answer": "Step 1: Identify the main topics in the dialogue.\n- Christmas party\n- Getting dressed\n- Formal party\n- Special party make up\n- Lipstick\n\nStep 2: Analyze the response.\n- The response mentions a rocket launch, which is not related to the main topics in the dialogue.\n- The response mentions bringing a gift, which could be related to the Christmas party.\n\nStep 3: Determine if the response can plausibly follow the dialogue.\nThe mention of a rocket launch seems out of context and unrelated to the dialogue. In addition the second part of the response mentions an obligation to bring a gift which wouldn't follow the first part as rocket launch event typically does not require to bring a gift. A plausible event would be a gift exchange event. \nSo the response does not plausibly follow the dialogue.\n\nFinal Answer: <Answer>No</Answer>"
            }
        ],
        "final_turn_prefix": "B:",
        "subdata_id": "3DW3BNF1HQ8B26ICDLVN1CZ2P3JV8X",
        "type": "bcq_cot"
    },
    {
        "data_id": "daily-dialogue-0001",
        "dialogue": [
            "A: ( Before Christmas Party ) Are you ready for the Christmas party tonight",
            "B: Almost. I have to get dressed. It's a formal party and I have special party make up!",
            "A: Use this lipstick and it will make your lips shine!"
        ],
        "final_turns": [
            {
                "id": "3X4JMASXCWZGXNDX322S20J3WU40BO-0",
                "knowledge": [
                    {
                        "dimension": "temporal",
                        "head": "gift exchange",
                        "head_turn": 4,
                        "id": "daily-dialogue-0001-5",
                        "relation": "HappensIn",
                        "tail": "Christmas party",
                        "tail_turn": 1,
                        "verbalized": "gift exchange happens in Christmas party"
                    }
                ],
                "label": 0,
                "text": "B: Great! Uh, remember that there's a rocket launch, too. We all have to bring a gift.",
                "votes": 1,
                "answer": "Step 1: Analyze the dialogue\nPerson A asks if Person B is ready for the Christmas party. Person B says they are almost ready and mentions the formal attire and makeup.\n\nStep 2: Analyze the knowledge\nThe knowledge states that gift exchanges typically happen at the Christmas parties.\n\nStep 3: Analyze the response\nPerson B responds positively to the lipstick suggestion and then mentions a rocket launch and the need to bring a gift.\n\nStep 4: Check for plausibility\nThe response seems to be a mix of two different ideas. The part about bringing a gift is plausible, as it relates to the knowledge about the gift exchange. However, the mention of a rocket launch seems unrelated and out of context. In addition, even if there was a rocket launch, it does not necessitate bringing a gift. So, the response overall is not plausible given this dialogue.\n\nFinal Answer: <Answer>No</Answer>",
            }
        ],
        "final_turn_prefix": "B:",
        "subdata_id": "3DW3BNF1HQ8B26ICDLVN1CZ2P3JV8X",
        "type": "bcq_cot_with_kg"
    }
]

BINARY_CHOICE_TEMPLATES = ["bcq", "bcq_with_kg", "bcq_cot", "bcq_cot_with_kg"]
MULTIPLE_CHOICE_TEMPLATES = ["mcq"]

def prepare_sample_for_bcq(sample, shot_samples, template):
    shots = []
    shot_ids = []

    for idx, shot_sample in enumerate(shot_samples):
        final_turn = [t for t in shot_sample["final_turns"] if t.get("knowledge")][0]
        knowledge = final_turn.get("knowledge")
        answer = "Yes" if final_turn["label"] == 1 else "No"

        if "answer" in final_turn:
            answer = final_turn["answer"]

        shot = SHOT_TEMPLATES[template].format(
            index=idx+1,
            dialogue="\n".join(prepare_dialogue_for_eval(shot_sample["dialogue"])),
            response=prepare_dialogue_turn_for_eval(final_turn["text"]),
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=answer,
        )
        shots.append(shot)
        shot_ids.append(final_turn["id"])

    shots_prompt = "\n\n".join(shots)

    eval_data = []

    for idx, final_turn in enumerate(sample["final_turns"]):
        # if final_turn["id"] in shot_ids:
        #     continue
        
        knowledge = final_turn.get("knowledge")
        final_shot = SHOT_TEMPLATES[template].format(
            index=len(shot_samples)+1,
            dialogue="\n".join(prepare_dialogue_for_eval(sample["dialogue"])),
            response=prepare_dialogue_turn_for_eval(final_turn["text"]),
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=""
        )
        prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
        eval_data.append({
            "data_id": sample["data_id"],
            "subdata_id": sample["subdata_id"],
            "instance_id": final_turn["id"],
            "dialogue": prepare_dialogue_for_eval(sample["dialogue"]),
            "final_turn_prefix": prepare_dialogue_turn_for_eval(sample["final_turn_prefix"]),
            "final_turn": {**final_turn, "text": prepare_dialogue_turn_for_eval(final_turn["text"])},
            "prompt": prompt,
            "answer": ("Yes" if final_turn["label"] == 1 else "No"),
            "type": template
        })

    return eval_data

def prepare_sample_for_mcq(sample, shot_samples, template):
    eval_data = []
    shots = []

    for s_idx, shot_sample in enumerate(shot_samples):
        final_turns = random.sample(shot_sample["final_turns"], len(shot_sample["final_turns"]))

        shot = SHOT_TEMPLATES[template].format(
            index=s_idx+1,
            dialogue="\n".join(prepare_dialogue_for_eval(shot_sample["dialogue"])),
            responses="\n".join([f"{i_idx+1}. {prepare_dialogue_turn_for_eval(turn['text'])}" for i_idx, turn in enumerate(final_turns)]),
            answer=",".join([str(i_idx+1) for i_idx, turn in enumerate(final_turns) if turn["label"] == 1])
        )
        shots.append(shot)

    shots_prompt = "\n\n".join(shots)

    final_turns = random.sample(sample["final_turns"], len(sample["final_turns"]))

    final_shot = SHOT_TEMPLATES[template].format(
        index=len(shot_samples)+1,
        dialogue="\n".join(prepare_dialogue_for_eval(sample["dialogue"])),
        responses="\n".join([f"{i_idx+1}. {prepare_dialogue_turn_for_eval(turn['text'])}" for i_idx, turn in enumerate(final_turns)]),
        answer=""
    )

    prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
    eval_data.append({
        "data_id": sample["data_id"],
        "subdata_id": sample["subdata_id"],
        "instance_id": sample["data_id"],
        "dialogue": prepare_dialogue_for_eval(sample["dialogue"]),
        "final_turn_prefix": prepare_dialogue_turn_for_eval(sample["final_turn_prefix"]),
        "final_turns": [{**final_turn, "text": prepare_dialogue_turn_for_eval(final_turn["text"])} for final_turn in final_turns],
        "prompt": prompt,
        "answer": ",".join([str(i_idx+1) for i_idx, turn in enumerate(final_turns) if turn["label"] == 1]),
        "type": template,
        "num_options": len(final_turns)
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