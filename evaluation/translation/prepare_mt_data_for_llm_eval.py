import sys
import argparse
import pathlib
from tqdm import tqdm
import pandas as pd
import random

sys.path.append("../..")

from utils import read_json, write_json

INSTRUCTION_TEMPLATES  = {
    "bcq": "You are a helpful assistant for translation from {source_lang} to {target_lang}. Given a sentence in {source_lang} and its translation in {target_lang}, answer whether the translation is correct. Answer only Yes or No.",
    "bcq_with_kg": "You are a helpful assistant for translation from {source_lang} to {target_lang}. Given a sentence in {source_lang}, its translation in {target_lang} and a relevant knowledge, answer whether the translation is correct. Answer only Yes or No.",
    "mcq": "You are a helpful assistant for translation from {source_lang} to {target_lang}. Given a sentence in {source_lang} and two translations of this sentence in {target_lang}, select the correct translation.",
    "bcq_cot": "You are a helpful assistant for translation from {source_lang} to {target_lang}. Given a sentence in {source_lang} and its translation in {target_lang}, answer whether the translation is correct. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>",
    "bcq_cot_with_kg": "You are a helpful assistant for translation from {source_lang} to {target_lang}. Given a sentence in {source_lang}, its translation in {target_lang} and a relevant knowledge, answer whether the translation is correct. Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags, <Answer>Yes/No</Answer>"
}

SHOT_TEMPLATES = {
    "bcq": "Example {index}:\n\nSentence ({source_lang}):\n{sentence}\n\nTranslation ({target_lang}):\n{translation}\n\nAnswer:{answer}",
    "bcq_with_kg": "Example {index}:\n\nSentence ({source_lang}):\n{sentence}\n\nTranslation ({target_lang}):\n{translation}\n\nKnowledge:\n{knowledge}\n\nAnswer:{answer}",
    "mcq": "Example {index}:\n\nSentence ({source_lang}):\n{sentence}\n\nTranslations:\n{translations}\n\nAnswer:{answer}",
    "bcq_cot": "Example {index}:\n\nSentence ({source_lang}):\n{sentence}\n\nTranslation ({target_lang}):\n{translation}\n\nAnswer:\n{answer}",
    "bcq_cot_with_kg": "Example {index}:\n\nSentence ({source_lang}):\n{sentence}\n\nTranslation ({target_lang}):\n{translation}\n\nKnowledge:\n{knowledge}\n\nAnswer:\n{answer}"
}

SHOT_SAMPLES = [
    {
        "data_id": "3IVKZBIBJ2NWN3SGA2DRQ31LSKRHSW-1",
        "subdata_id": "3IVKZBIBJ2NWN3SGA2DRQ31LSKRHSW-1",
        "source_lang": "en",
        "target_lang": "de",
        "source": "The song took longer to sing than the ballad because it was more words.",
        "targets": [
            {
                "text": "Das Lied brauchte l\u00e4nger zum Singen als die Ballade, weil sie mehr Worte enthielt.",
                "label": 0,
                "answer": "Let's break down the sentence and its translation:\n\n- The song: Das Lied\n- took longer: brauchte länger\n- to sing: zum Singen\n- than: als\n- the ballad: die Ballade\n- because: weil\n- it: sie\n- was more words: mehr Worte enthielt (literally: contained more words)\n\nThe translation is almost correct, but the pronoun for 'it' should be 'es' instead of 'sie'. This is because 'it' should refer to the thing that has more words and since 'song' took longer to sing than 'ballad', 'song' should contain more words, and the correct pronoun for 'song' is 'es', not 'sie'. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3TXMY6UCAOENVLA2Y0S2SQG3UXIQCU-0",
                "head": "more words",
                "relation": "Causes",
                "tail": "longer",
                "dimension": "causal",
                "verbalized": "more words causes longer"
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "3I7SHAD35MUH2UASTYJTVJPDY547MI-2",
        "subdata_id": "3I7SHAD35MUH2UASTYJTVJPDY547MI-2",
        "source_lang": "en",
        "target_lang": "fr",
        "source": "Bob would rather fill his emergency fund using his mobile instead of the bank because it was handy.",
        "targets": [
            {
                "text": "Bob pr\u00e9f\u00e9rait remplir son fonds d' urgence en utilisant son mobile plut\u00f4t qu' avec la banque car elle \u00e9tait \u00e0 port\u00e9e de main.",
                "label": 0,
                "answer": "Let's break down the sentence and compare each part:\n\n1. Bob would rather fill his emergency fund\n   Bob préférait remplir son fonds d'urgence\n\n2. using his mobile instead of the bank\n   en utilisant son mobile plutôt qu'avec la banque\n\n3. because it was handy.\n   car elle était à portée de main.\n\nThe translation seems accurate, but there is a small issue with the gender of 'it' in the last part. In the original sentence, 'it' refers to Bob's 'mobile' because he would prefer using it, so it is handy, but in French, 'mobile' is masculine, so 'elle' should be replaced with 'il'. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3483FV8BEO9HNLZGMYPYA8QQI7L26W-0",
                "head": "rather fill his emergency fund",
                "relation": "Implies",
                "tail": "handy",
                "dimension": "causal",
                "verbalized": "rather fill his emergency fund implies handy"
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "3PKVGQTFIJY68JIS5DHAMTRTXHGYRS-1",
        "subdata_id": "3PKVGQTFIJY68JIS5DHAMTRTXHGYRS-1",
        "source_lang": "en",
        "target_lang": "ru",
        "source": "Joe got into the school but not into the University after hours because it was unlocked and insecure.",
        "targets": [
            {
                "text": "\u0414\u0436\u043e \u043f\u043e\u043f\u0430\u043b \u0432 \u0448\u043a\u043e\u043b\u0443, \u043d\u043e \u043d\u0435 \u0432 \u0443\u043d\u0438\u0432\u0435\u0440\u0441\u0438\u0442\u0435\u0442 \u0432 \u043d\u0435\u0440\u0430\u0431\u043e\u0447\u0435\u0435 \u0432\u0440\u0435\u043c\u044f, \u043f\u043e\u0442\u043e\u043c\u0443 \u0447\u0442\u043e o\u043d \u0431\u044b\u043b \u043d\u0435\u0437\u0430\u0449\u0438\u0449\u0435\u043d\u043d\u044b\u043c \u0438 \u043d\u0435\u0431\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u044b\u043c.",
                "label": 0,
                "answer": "Let's break down the sentence and translation:\n\nJoe - Джо\ngot into the school - попал в школу\nbut not into the University - но не в университет\nafter hours - в нерабочее время\nbecause - потому что\nit was unlocked - oн был незащищенным\nand insecure - и небезопасным\n\nThe translation is almost correct, but the phrase 'it was unlocked' is not translated accurately. This is because in the source sentence 'it' refers to something 'insecure' and 'unlocked' so Joe could get into and since he got into school, 'it' should refer to 'school', but in the translation, 'it' refers to 'university' which is wrong. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3VSOLARPKLZTQMXXO56FDSMKEMU93Q-0",
                "head": "unlocked and insecure",
                "relation": "Causes",
                "tail": "get into after hours",
                "dimension": "causal",
                "verbalized": "unlocked and insecure causes get into after hours"
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "chinese-mt-la-350",
        "subdata_id": "chinese-mt-la-350",
        "source_lang": "zh",
        "target_lang": "en",
        "source": "\u4ed6\u6b63\u5728\u7f16\u660e\u5929\u665a\u4f1a\u4e0a\u8981\u5531\u7684\u6b4c\u7684\u6b4c\u8bcd\u3002",
        "targets": [
            {
                "text": "He is weaving the lyrics of the song to be sung at the party tomorrow.",
                "label": 0,
                "answer": "Let's break down the Chinese sentence:\n\n他 (He) 正在 (is) 编 (weaving/writing) 明天 (tomorrow) 晚会 (party) 上要 (at) 唱 (sing) 的歌 (the song) 的歌词 (lyrics)。\n\nThe correct translation should be:\nHe is writing the lyrics of the song to be sung at the party tomorrow.\n\nThe given translation uses 'weaving' instead of 'writing', but lyrics can not be woven, instead they can be written. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3JPSL1DZ52PEZ90CYU4UI1VAQ1UANA-0",
                "head": "lyrics",
                "relation": "Not CapableOf",
                "tail": "being woven",
                "votes": 3,
                "dimension": "attribution",
                "verbalized": "lyrics is not capable of being woven"
            }
        ],
        "type": "bcq_cot"
    },
    {
        "data_id": "3IVKZBIBJ2NWN3SGA2DRQ31LSKRHSW-1",
        "subdata_id": "3IVKZBIBJ2NWN3SGA2DRQ31LSKRHSW-1",
        "source_lang": "en",
        "target_lang": "de",
        "source": "The song took longer to sing than the ballad because it was more words.",
        "targets": [
            {
                "text": "Das Lied brauchte l\u00e4nger zum Singen als die Ballade, weil sie mehr Worte enthielt.",
                "label": 0,
                "answer": "Let's break down the sentence and its translation:\n\n- The song: Das Lied\n- took longer: brauchte länger\n- to sing: zum Singen\n- than: als\n- the ballad: die Ballade\n- because: weil\n- it: sie\n- was more words: mehr Worte enthielt (literally: contained more words)\n\nThe translation is almost correct, but the pronoun for 'it' should be 'es' instead of 'sie'. This is because 'it' should refer to the thing that has more words and since 'song' took longer to sing than 'ballad' and according to the given knowledge, more words implies longer singing, so 'song' should contain more words, and the correct pronoun for 'song' is 'es', not 'sie'. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3TXMY6UCAOENVLA2Y0S2SQG3UXIQCU-0",
                "head": "more words",
                "relation": "Causes",
                "tail": "longer",
                "dimension": "causal",
                "verbalized": "more words causes longer"
            }
        ],
        "type": "bcq_cot_with_kg"
    },
    {
        "data_id": "3I7SHAD35MUH2UASTYJTVJPDY547MI-2",
        "subdata_id": "3I7SHAD35MUH2UASTYJTVJPDY547MI-2",
        "source_lang": "en",
        "target_lang": "fr",
        "source": "Bob would rather fill his emergency fund using his mobile instead of the bank because it was handy.",
        "targets": [
            {
                "text": "Bob pr\u00e9f\u00e9rait remplir son fonds d' urgence en utilisant son mobile plut\u00f4t qu' avec la banque car elle \u00e9tait \u00e0 port\u00e9e de main.",
                "label": 0,
                "answer": "Let's break down the sentence and compare each part:\n\n1. Bob would rather fill his emergency fund\n   Bob préférait remplir son fonds d'urgence\n\n2. using his mobile instead of the bank\n   en utilisant son mobile plutôt qu'avec la banque\n\n3. because it was handy.\n   car elle était à portée de main.\n\nThe translation seems accurate, but there is a small issue with the gender of 'it' in the last part. In the original sentence, 'it' refers to Bob's 'mobile' because he would prefer using it and according to the knowledge, this would imply being handy, but in French, 'mobile' is masculine, so 'elle' should be replaced with 'il'. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3483FV8BEO9HNLZGMYPYA8QQI7L26W-0",
                "head": "rather fill his emergency fund",
                "relation": "Implies",
                "tail": "handy",
                "dimension": "causal",
                "verbalized": "rather fill his emergency fund implies handy"
            }
        ],
        "type": "bcq_cot_with_kg"
    },
    {
        "data_id": "3PKVGQTFIJY68JIS5DHAMTRTXHGYRS-1",
        "subdata_id": "3PKVGQTFIJY68JIS5DHAMTRTXHGYRS-1",
        "source_lang": "en",
        "target_lang": "ru",
        "source": "Joe got into the school but not into the University after hours because it was unlocked and insecure.",
        "targets": [
            {
                "text": "\u0414\u0436\u043e \u043f\u043e\u043f\u0430\u043b \u0432 \u0448\u043a\u043e\u043b\u0443, \u043d\u043e \u043d\u0435 \u0432 \u0443\u043d\u0438\u0432\u0435\u0440\u0441\u0438\u0442\u0435\u0442 \u0432 \u043d\u0435\u0440\u0430\u0431\u043e\u0447\u0435\u0435 \u0432\u0440\u0435\u043c\u044f, \u043f\u043e\u0442\u043e\u043c\u0443 \u0447\u0442\u043e o\u043d \u0431\u044b\u043b \u043d\u0435\u0437\u0430\u0449\u0438\u0449\u0435\u043d\u043d\u044b\u043c \u0438 \u043d\u0435\u0431\u0435\u0437\u043e\u043f\u0430\u0441\u043d\u044b\u043c.",
                "label": 0,
                "answer": "Let's break down the sentence and translation:\n\nJoe - Джо\ngot into the school - попал в школу\nbut not into the University - но не в университет\nafter hours - в нерабочее время\nbecause - потому что\nit was unlocked - oн был незащищенным\nand insecure - и небезопасным\n\nThe translation is almost correct, but the phrase 'it was unlocked' is not translated accurately. This is because in the source sentence 'it' refers to something 'insecure' and 'unlocked' and according to the knowledge, insecure and unlocked means it is easy to get into and since he got into school, 'it' should refer to 'school', but in the translation, 'it' refers to 'university' which is wrong. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3VSOLARPKLZTQMXXO56FDSMKEMU93Q-0",
                "head": "unlocked and insecure",
                "relation": "Causes",
                "tail": "get into after hours",
                "dimension": "causal",
                "verbalized": "unlocked and insecure causes get into after hours"
            }
        ],
        "type": "bcq_cot_with_kg"
    },
    {
        "data_id": "chinese-mt-la-350",
        "subdata_id": "chinese-mt-la-350",
        "source_lang": "zh",
        "target_lang": "en",
        "source": "\u4ed6\u6b63\u5728\u7f16\u660e\u5929\u665a\u4f1a\u4e0a\u8981\u5531\u7684\u6b4c\u7684\u6b4c\u8bcd\u3002",
        "targets": [
            {
                "text": "He is weaving the lyrics of the song to be sung at the party tomorrow.",
                "label": 0,
                "answer": "Let's break down the Chinese sentence:\n\n他 (He) 正在 (is) 编 (weaving/writing) 明天 (tomorrow) 晚会 (party) 上要 (at) 唱 (sing) 的歌 (the song) 的歌词 (lyrics)。\n\nThe correct translation should be:\nHe is writing the lyrics of the song to be sung at the party tomorrow.\n\nThe given translation uses 'weaving' instead of 'writing', but as given in the knowledge, lyrics can not be woven, instead they can be written. So the translation is not correct.\n\n<Answer>No</Answer>"
            }
        ],
        "knowledge": [
            {
                "id": "3JPSL1DZ52PEZ90CYU4UI1VAQ1UANA-0",
                "head": "lyrics",
                "relation": "Not CapableOf",
                "tail": "being woven",
                "votes": 3,
                "dimension": "attribution",
                "verbalized": "lyrics is not capable of being woven"
            }
        ],
        "type": "bcq_cot_with_kg"
    }
]

BINARY_CHOICE_TEMPLATES = ["bcq", "bcq_with_kg", "bcq_cot", "bcq_cot_with_kg"]
MULTIPLE_CHOICE_TEMPLATES = ["mcq"]

LANGUAGE_MAP = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "zh": "Chinese",
}

def prepare_sample_for_bcq(sample, shot_samples, template, source_lang, target_lang):
    shots = []
    shot_ids = []

    for idx, shot_sample in enumerate(shot_samples):
        translation = shot_sample["targets"][0]
        knowledge = shot_sample.get("knowledge")
        answer = "Yes" if translation["label"] == 1 else "No"

        if "answer" in translation:
            answer = translation["answer"]

        shot = SHOT_TEMPLATES[template].format(
            index=idx+1,
            sentence=shot_sample["source"],
            source_lang=LANGUAGE_MAP[source_lang],
            target_lang=LANGUAGE_MAP[target_lang],
            translation=translation["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=answer
        )
        shots.append(shot)
        shot_ids.append(f"{shot_sample['data_id']}-{translation['label']}")

    shots_prompt = "\n\n".join(shots)

    eval_data = []

    for idx, translation in enumerate(sample["targets"]):
        instance_id = f"{sample['data_id']}-{translation['label']}"
        
        # if instance_id in shot_ids:
        #     continue

        knowledge = sample.get("knowledge")
        final_shot = SHOT_TEMPLATES[template].format(
            index=len(shot_samples)+1,
            sentence=sample["source"],
            source_lang=LANGUAGE_MAP[source_lang],
            target_lang=LANGUAGE_MAP[target_lang],
            translation=translation["text"],
            knowledge="\n".join([kg["verbalized"] for kg in knowledge]) if knowledge else "None",
            answer=""
        )
        instruction = INSTRUCTION_TEMPLATES[template].format(source_lang=LANGUAGE_MAP[source_lang], target_lang=LANGUAGE_MAP[target_lang])
        prompt = f"{instruction}\n\n{shots_prompt}\n\n{final_shot}"
        eval_data.append({
            "data_id": sample["data_id"],
            "subdata_id": sample["data_id"],
            "instance_id": instance_id,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "sentence": sample["source"],
            "translation": {**translation, "knowledge": sample["knowledge"]},
            "prompt": prompt,
            "answer": ("Yes" if translation["label"] == 1 else "No"),
            "type": template
        })

    return eval_data

def prepare_sample_for_mcq(sample, shot_samples, template, source_lang, target_lang):
    eval_data = []
    shots = []

    for s_idx, shot_sample in enumerate(shot_samples):
        translations = random.sample(shot_sample["translations"], len(shot_sample["translations"]))

        shot = SHOT_TEMPLATES[template].format(
            index=s_idx+1,
            sentence=shot_sample["source"],
            source_lang=LANGUAGE_MAP[source_lang],
            target_lang=LANGUAGE_MAP[target_lang],
            translations="\n".join([f"{i_idx+1}. {translation['text']}" for i_idx, translation in enumerate(translations)]),
            answer=",".join([str(i_idx+1) for i_idx, translation in enumerate(translations) if translation["label"] == 1])
        )
        shots.append(shot)

    shots_prompt = "\n\n".join(shots)

    translations = random.sample(sample["translations"], len(sample["translations"]))

    final_shot = SHOT_TEMPLATES[template].format(
        index=len(shot_samples)+1,
        sentence=sample["source"],
        translations="\n".join([f"{i_idx+1}. {translation['text']}" for i_idx, translation in enumerate(translations)]),
        answer=""
    )

    instruction = INSTRUCTION_TEMPLATES[template].format(source_lang=LANGUAGE_MAP[source_lang], target_lang=LANGUAGE_MAP[target_lang])
    prompt = f"{instruction}\n\n{shots_prompt}\n\n{final_shot}"
    eval_data.append({
        "data_id": sample["data_id"],
        "subdata_id": sample["data_id"],
        "source_lang": source_lang,
        "target_lang": target_lang,
        "sentence": sample["source"],
        "translations": translations,
        "prompt": prompt,
        "answer": ",".join([str(i_idx+1) for i_idx, translation in enumerate(translations) if translation["label"] == 1]),
        "type": "mcq",
        "num_options": len(translations)
    })
    
    return eval_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to eval data in json", required=True)
    parser.add_argument("--template", type=str, default="bcq")
    parser.add_argument("--num-shots", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="", help="Custom suffix for output file path.")
    parser.add_argument("--source-lang", type=str, default="en")
    parser.add_argument("--target-lang", type=str, default="de")

    args = parser.parse_args()
    results = read_json(args.datapath)

    eval_data = []

    shot_samples = [sample for sample in SHOT_SAMPLES if sample["type"] == args.template and sample["source_lang"] == args.source_lang and sample["target_lang"] == args.target_lang][:args.num_shots]

    if not shot_samples:
        shot_samples = results[:args.num_shots]

    for result in tqdm(results, desc="Preparing results for LLM evaluation"):
        if args.template in BINARY_CHOICE_TEMPLATES:
            eval_data.extend(prepare_sample_for_bcq(result, shot_samples, args.template, args.source_lang, args.target_lang))
        elif args.template in MULTIPLE_CHOICE_TEMPLATES:
            eval_data.extend(prepare_sample_for_mcq(result, shot_samples, args.template, args.source_lang, args.target_lang))
        else:
            raise ValueError(f"Invalid template: {args.template}")

    
    datapath = pathlib.Path(args.datapath)
    eval_data_path_stem = datapath.parent / f"{datapath.stem}_llm_{args.template}{args.suffix}"

    write_json(eval_data, eval_data_path_stem.with_suffix(".json"))
    pd.DataFrame(eval_data).to_csv(eval_data_path_stem.with_suffix(".csv"), index=False)

if __name__ == "__main__":
    main()