import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import random

sys.path.append("../..")

from utils import read_json, write_json, prepare_dialogue_for_eval

INSTRUCTION_TEMPLATES  = {
    "bcq": "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2#, answer whether the given summary correctly summarizes the dialogue. Answer only 'Yes' or 'No'.",
    "bcq_with_kg": "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2# and a relevant knowledge, answer whether the given summary correctly summarizes the dialogue. Answer only 'Yes' or 'No'.",
    "mcq": "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2#, select summaries that correctly summarize this dialogue. Select all that apply.",
    "bcq_cot": "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2#, answer whether the given summary correctly summarizes the dialogue. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>",
    "bcq_cot_with_kg": "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2# and a relevant knowledge, answer whether the given summary correctly summarizes the dialogue. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>"
}

SHOT_TEMPLATES = {
    "bcq": "Example {index}:\n\nDialogue:\n{dialogue}\n\nSummary:\n{summary}\n\nAnswer:{answer}",
    "bcq_with_kg": "Example {index}:\n\nDialogue:\n{dialogue}\n\nSummary:\n{summary}\n\nKnowledge:\n{knowledge}\n\nAnswer:{answer}",
    "mcq": "Example {index}:\n\nDialogue:\n{dialogue}\n\nSummaries:\n{summaries}\n\nAnswer:{answer}",
    "bcq_cot": "Example {index}:\n\nDialogue:\n{dialogue}\n\nSummary:\n{summary}\n\nAnswer:\n{answer}",
    "bcq_cot_with_kg": "Example {index}:\n\nDialogue:\n{dialogue}\n\nSummary:\n{summary}\n\nKnowledge:\n{knowledge}\n\nAnswer:\n{answer}"
}

SHOT_SAMPLES = [
    {
        "data_id": "dialogsum_test_77",
        "subdata_id": "372AGES0JDV9O023C98OMAGT156XR6",
        "dialogue": [
            "#Person1#: I'm going to New York for the first time, but I don't have a tour guide. Can you give me any suggestions?",
            "#Person2#: There's a service called 'A friend in New York'. It's a personal tour guide service.",
            "#Person1#: That's interesting. What does it do?",
            "#Person2#: You give them your information by answering a questionnaire and they will create a perfect trip for you according to your budget.",
            "#Person1#: Good. Where can I get the questionnaire?",
            "#Person2#: You can easily download it from their website.",
            "#Person1#: That's helpful! Thanks!"
        ],
        "summaries": [
            {
                "id": "3F6KKYWMNLRCXG3OI4VW823G9LDDNX-1",
                "text": "#Person1# is going to New York for the first time. #Person2# suggests #Person1# use a personal tour guide service even though they won't know how to put together #Person1#'s trip plan.",
                "votes": 2,
                "label": 0,
                "knowledge": [
                    {
                        "id": "39U1BHVTDVHCA16BMBBW4SNW7PU3T8-0",
                        "head": "personal tour guide service",
                        "relation": "CapableOf",
                        "tail": "make #Person1#'s trip plan.",
                        "head_from": "dialogue",
                        "tail_from": "summary",
                        "dimension": "attribution",
                        "verbalized": "personal tour guide service is capable of make #Person1#'s trip plan."
                    }
                ],
                "answer": "Step 1: Identify the main points in the dialogue.\n- #Person1# is going to New York for the first time and needs suggestions.\n- #Person2# suggests 'A friend in New York' service.\n- The service creates a perfect trip based on a questionnaire.\n- The questionnaire can be downloaded from their website.\n\nStep 2: Compare the summary with the main points.\n- The summary correctly mentions that #Person1# is going to New York for the first time.\n- The summary mentions the personal tour guide service, but it incorrectly states that they won't know how to put together #Person1#'s trip plan because according the dialogue, the service can create a perfect trip based on the questionnaire.\n\nFinal Answer: <Answer>No</Answer>",
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "dialogsum_test_49",
        "subdata_id": "3LN50BUKQ41TCT5ZLY1B6LLRTE8PLK",
        "dialogue": [
            "#Person1#: OK, that's a cut. Let's start from the beginning everyone.",
            "#Person2#: What was the problem that time?",
            "#Person1#: The feeling was all wrong, Mike. She is telling you that she doesn't want to see you anymore, but I want to get more anger from you. You're acting hurt and sad, but that's not how your character would act in this situation.",
            "#Person2#: But Jason and Laura had been together for 3 years. Don't you think his reaction would be one of both anger and sadness?",
            "#Person1#: At this point, no. I think he would react the way most guys would and then later on, we would see his real feelings.",
            "#Person2#: I'm not so sure about that.",
            "#Person1#: Let's try it my way and you can see how you feel when you're saying your lines. After that, if it still doesn't feel right we can try something else."
        ],
        "summaries": [
            {
                "id": "3X3OR7WPZ9QASL4CPJ419ILQT5YL8R-1",
                "text": "#Person1# and Mike are discussing what kind of emotion should be expressed by Mike in this play. They have different understandings of the definition of anger",
                "label": 0,
                "knowledge": [
                    {
                        "id": "3DOCMVPBTX4LJSU17Y9Z47FANOWNN4-0",
                        "head": "Don't you think his reaction would be one of both anger and sadness?",
                        "relation": "Implies",
                        "tail": "They have different understandings.",
                        "head_from": "dialogue",
                        "tail_from": "summary",
                        "dimension": "causal",
                        "verbalized": "Don't you think his reaction would be one of both anger and sadness? implies They have different understandings."
                    }
                ],
                "answer": "Step 1: Identify the main points of the dialogue\n- #Person1# wants Mike to express more anger in his acting\n- Mike thinks his character should show both anger and sadness\n- #Person1# suggests trying his way first, and if it doesn't feel right, they can try something else\n\nStep 2: Identify the main points covered in the summary\n- The summary mentions that #Person1# and Mike are discussing the emotion to be expressed in the play\n- The summary states that they have different understandings of the definition of anger\n\nStep 3: Compare the summary with the dialogue and knowledge\n- The knowledge states that they have different understandings of the character's emotions while the given summary points out a different understanding of the definition of anger. These are not the same things.\n- Dialogue is about deciding which emotion to express for the character, not about the definition of anger.\n\nStep 4: Determine if the summary is correct\n- The summary does not correctly summarize the dialogue because it misinterprets the conflict between the #Person1# and #Person2#.\n\n<Answer>No</Answer>",
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
        summary = [s for s in shot_sample["summaries"] if s.get("knowledge")][0]
        knowledge = summary.get("knowledge")
        answer = "Yes" if summary["label"] == 1 else "No"

        if "answer" in summary:
            answer = summary["answer"]

        shot = SHOT_TEMPLATES[template].format(
            index=idx+1,
            dialogue="\n".join(prepare_dialogue_for_eval(shot_sample["dialogue"])),
            summary=summary["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=answer
        )
        shots.append(shot)
        shot_ids.append(summary["id"])

    shots_prompt = "\n\n".join(shots)

    eval_data = []

    for idx, summary in enumerate(sample["summaries"]):
        # if summary["id"] in shot_ids:
        #     continue

        knowledge = summary.get("knowledge")
        final_shot = SHOT_TEMPLATES[template].format(
            index=len(shot_samples)+1,
            dialogue="\n".join(prepare_dialogue_for_eval(sample["dialogue"])),
            summary=summary["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=""
        )
        prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
        eval_data.append({
            "data_id": sample["data_id"],
            "subdata_id": sample["subdata_id"],
            "instance_id": summary["id"],
            "dialogue": prepare_dialogue_for_eval(sample["dialogue"]),
            "summary": summary,
            "prompt": prompt,
            "answer": ("Yes" if summary["label"] == 1 else "No"),
            "type": template
        })

    return eval_data

def prepare_sample_for_mcq(sample, shot_samples, template):
    eval_data = []
    shots = []

    for s_idx, shot_sample in enumerate(shot_samples):
        summaries = random.sample(shot_sample["summaries"], len(shot_sample["summaries"]))

        shot = SHOT_TEMPLATES[template].format(
            index=s_idx+1,
            dialogue="\n".join(prepare_dialogue_for_eval(shot_sample["dialogue"])),
            summaries="\n".join([f"{i_idx+1}. {summary['text']}" for i_idx, summary in enumerate(summaries)]),
            answer=",".join([str(i_idx+1) for i_idx, summary in enumerate(summaries) if summary["label"] == 1])
        )
        shots.append(shot)

    shots_prompt = "\n\n".join(shots)

    summaries = random.sample(sample["summaries"], len(sample["summaries"]))

    final_shot = SHOT_TEMPLATES[template].format(
        index=len(shot_samples)+1,
        dialogue="\n".join(prepare_dialogue_for_eval(sample["dialogue"])),
        summaries="\n".join([f"{i_idx+1}. {summary['text']}" for i_idx, summary in enumerate(summaries)]),
        answer=""
    )

    prompt = f"{INSTRUCTION_TEMPLATES[template]}\n\n{shots_prompt}\n\n{final_shot}"
    eval_data.append({
        "data_id": sample["data_id"],
        "subdata_id": sample["subdata_id"],
        "instance_id": sample["data_id"],
        "dialogue": prepare_dialogue_for_eval(sample["dialogue"]),
        "summaries": summaries,
        "prompt": prompt,
        "answer": ",".join([str(i_idx+1) for i_idx, summary in enumerate(summaries) if summary["label"] == 1]),
        "type": template,
        "num_options": len(summaries)
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