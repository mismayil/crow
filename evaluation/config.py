TASKS = ["mt_zh_en", "mt_en_fr", "mt_en_de", "mt_en_ru", "dialogue", "summarization", "stance", "safety", "intent"]
LEADERBOARD_PATH = "data/leaderboard.json"
TASK_LEADERBOARD_PATH = "data/task_leaderboard.json"
REFERENCE_PATH = "../data/private/crow_{task}_private.json"
INVALID_RESULTS_DIR = "data/invalid"
SUBMISSIONS_PATH = "data/submissions.json"
SUBMISSIONS_DIR = "data/submissions"
INTERNAL_SUBMISSIONS_PATH = "data/internal_submissions.csv"
SUBMISSIONS_COLUMNS = ["Timestamp", "Username", "Contributor", "Model Name", "Model Size", "Model Link", "Task(s)", "Predictions", "Comments"]

TASK_NAME_MAP = {
    "Dialogue": "dialogue",
    "Dialogue Summarization": "summarization",
    "Intent Detection": "intent",
    "Safety Detection": "safety",
    "Stance Classification": "stance",
    "Machine Translation (en-de)": "mt_en_de",
    "Machine Translation (en-fr)": "mt_en_fr",
    "Machine Translation (en-ru)": "mt_en_ru",
    "Machine Translation (zh-en)": "mt_zh_en"
}

TASK_MAP = {
    "dialogue": {
        "context_field": "dialogue",
        "target_field": "response",
        "context": ["dialogue"],
        "target": ["final_turn", "text"],
        "knowledge": ["final_turn", "knowledge"],
        "label": ["final_turn", "label"]
    },
    "summarization": {
        "context_field": "dialogue",
        "target_field": "summary",
        "context": ["dialogue"],
        "target": ["summary", "text"],
        "knowledge": ["summary", "knowledge"],
        "label": ["summary", "label"]
    },
    "intent": {
        "context_field": "headline",
        "target_field": "intent",
        "context": ["headline"],
        "target": ["intent", "text"],
        "knowledge": ["intent", "knowledge"],
        "label": ["intent", "label"]
    },
    "safety": {
        "context_field": "scenario",
        "target_field": "action",
        "context": ["scenario"],
        "target": ["action", "text"],
        "knowledge": ["action", "knowledge"],
        "label": ["action", "label"],
        "label_to_id": {
            "safe": 1,
            "unsafe": 0
        }
    },
    "stance": {
        "context_field": "belief",
        "target_field": "argument",
        "context": ["instance", "belief"],
        "target": ["instance", "argument"],
        "knowledge": ["instance", "knowledge"],
        "label": ["instance", "stance"],
        "label_to_id": {
            "support": 1,
            "counter": 0
        }
    },
    "mt_en_fr": {
        "context_field": "sentence",
        "target_field": "translation",
        "context": ["sentence"],
        "target": ["translation", "text"],
        "knowledge": ["translation", "knowledge"],
        "label": ["translation", "label"],
        "extra_fields": ["source_lang", "target_lang"]
    },
    "mt_en_de": {
        "context_field": "sentence",
        "target_field": "translation",
        "context": ["sentence"],
        "target": ["translation", "text"],
        "knowledge": ["translation", "knowledge"],
        "label": ["translation", "label"],
        "extra_fields": ["source_lang", "target_lang"]
    },
    "mt_en_ru": {
        "context_field": "sentence",
        "target_field": "translation",
        "context": ["sentence"],
        "target": ["translation", "text"],
        "knowledge": ["translation", "knowledge"],
        "label": ["translation", "label"],
        "extra_fields": ["source_lang", "target_lang"]
    },
    "mt_zh_en": {
        "context_field": "sentence",
        "target_field": "translation",
        "context": ["sentence"],
        "target": ["translation", "text"],
        "knowledge": ["translation", "knowledge"],
        "label": ["translation", "label"],
        "extra_fields": ["source_lang", "target_lang"]
    }
}

MODEL_MAP = {
    "random": {
        "name": "Random Baseline",
        "size": "0",
        "link": ""
    },
    "majority": {
        "name": "Majority Baseline",
        "size": "0",
        "link": ""
    },
    "llama7": {
        "name": "Llama 1",
        "size": "7B",
        "link": "https://github.com/facebookresearch/llama"
    },
    "llama13": {
        "name": "Llama 1",
        "size": "13B",
        "link": "https://github.com/facebookresearch/llama"
    },
    "llama33": {
        "name": "Llama 1",
        "size": "33B",
        "link": "https://github.com/facebookresearch/llama"
    },
    "alpaca": {
        "name": "Alpaca",
        "size": "7B",
        "link": "https://crfm.stanford.edu/2023/03/13/alpaca.html"
    },
    "flan-t5": {
        "name": "Flan-T5",
        "size": "11B",
        "link": "https://huggingface.co/google/flan-t5-xxl"
    },
    "flan-alpaca": {
        "name": "Flan-Alpaca",
        "size": "3B",
        "link": "https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl"
    },
    "vicuna": {
        "name": "Vicuna",
        "size": "13B",
        "link": "https://lmsys.org/blog/2023-03-30-vicuna"
    },
    "stable-vicuna": {
        "name": "Stable-Vicuna",
        "size": "13B",
        "link": "https://huggingface.co/CarperAI/stable-vicuna-13b-delta"
    },
    "mt0": {
        "name": "mt0",
        "size": "13B",
        "link": "https://huggingface.co/bigscience/mt0-xxl"
    },
    "bloomz7": {
        "name": "BloomZ",
        "size": "7B",
        "link": "https://huggingface.co/bigscience/bloomz-7b1"
    },
    "palm": {
        "name": "Palm 1",
        "size": "540B",
        "link": "https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html"
    },
    "text-davinci-003": {
        "name": "GPT-3",
        "size": "175B",
        "link": "https://platform.openai.com/docs/models/gpt-3"
    },
    "gpt-4": {
        "name": "GPT-4",
        "size": "unknown",
        "link": "https://openai.com/research/gpt-4"
    },
    "gpt-4-cot": {
        "name": "GPT-4-CoT",
        "size": "unknown",
        "link": "https://openai.com/research/gpt-4"
    },
    "gpt-4+with-kg": {
        "name": "GPT-4+Oracle KG",
        "size": "unknown",
        "link": "https://openai.com/research/gpt-4"
    },
    "gpt-4-cot+with-kg": {
        "name": "GPT-4-CoT+Oracle KG",
        "size": "unknown",
        "link": "https://openai.com/research/gpt-4"
    },
}

MODELS = ["majority", "random", "llama7", "llama13", "llama33", "flan-t5", "alpaca", "flan-alpaca", "vicuna", "stable-vicuna", "mt0", "bloomz7", "palm", "text-davinci-003", "gpt-4"]

HUMAN_METRICS = {
    "mt_zh_en": {
        "macro_f1": 0.879,
        "exact_match": 0.78
    },
    "mt_en_fr": {
        "macro_f1": 0.83,
        "exact_match": 0.829
    },
    "mt_en_de": {
        "macro_f1": 0.899,
        "exact_match": 0.82
    },
    "mt_en_ru": {
        "macro_f1": 0.899,
        "exact_match": 0.86
    },
    "dialogue": {
        "macro_f1": 0.87,
        "exact_match": 0.869
    },
    "summarization": {
        "macro_f1": 0.989,
        "exact_match": 0.964
    },
    "stance": {
        "macro_f1": 0.881,
        "exact_match": 0.696
    },
    "safety": {
        "macro_f1": 0.978,
        "exact_match": 0.939
    },
    "intent": {
        "macro_f1": 0.939,
        "exact_match": 0.807
    }
}