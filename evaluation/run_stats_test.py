import argparse
import sys
import pprint
import pandas as pd
import scipy.stats as stats
import random
from tqdm import tqdm
import pathlib

sys.path.append("../")

from utils import read_json, write_json, find_json_files

TASK_MAP = {
    "dialogue": {
        "human_id_attr": "response_id",
        "model_id_attr": "instance_id",
        "model_target_attr": "final_turn",
        "human_target_attr": "response"
    },
    "summarization": {
        "human_id_attr": "summary_id",
        "model_id_attr": "instance_id",
        "model_target_attr": "summary",
        "human_target_attr": "summary"
    },
    "intent": {
        "human_id_attr": "intent_id",
        "model_id_attr": "instance_id",
        "model_target_attr": "intent",
        "human_target_attr": "intent"
    },
    "stance": {
        "human_id_attr": "instance_id",
        "model_id_attr": "instance_id",
        "model_target_attr": "instance",
        "model_target_subattr": ["belief", "argument"],
        "human_target_attr": ["belief", "argument"],
        "no_text": True
    },
    "safety": {
        "human_id_attr": "action_id",
        "model_id_attr": "instance_id",
        "model_target_attr": "action",
        "human_target_attr": "action"
    },
    "mt_zh_en": {
        "human_id_attr": "translation_id",
        "model_id_attr": "instance_id",
        "target_attr": "translation",
        "human_target_attr": "translation"
    },
    "mt_en_de": {
        "human_id_attr": "translation_id",
        "model_id_attr": "instance_id",
        "target_attr": "translation",
        "human_target_attr": "translation"
    },
    "mt_en_fr": {
        "human_id_attr": "translation_id",
        "model_id_attr": "instance_id",
        "target_attr": "translation",
        "human_target_attr": "translation"
    },
    "mt_en_ru": {
        "human_id_attr": "translation_id",
        "model_id_attr": "instance_id",
        "target_attr": "translation",
        "human_target_attr": "translation"
    }
}

def computer_mcnemar(df, sig_level=0.01):
    # create contingency table
    data_crosstab = pd.crosstab(df['model'],
                                df['human'],
                                margins=True, margins_name="Total")
    pprint.pprint(data_crosstab)

    # Calcualtion of McNemar's statistic
    rows = df['model'].unique()
    columns = df['human'].unique()
    mcnemar = (abs(data_crosstab['correct']['incorrect'] - data_crosstab['incorrect']['correct']) - 1)**2 / (data_crosstab['correct']['incorrect'] + data_crosstab['incorrect']['correct'])


    # The p-value approach
    print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
    p_value = 1 - stats.chi2.cdf(mcnemar, (len(rows)-1)*(len(columns)-1))
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= sig_level:
        conclusion = "Null Hypothesis is rejected."
            
    print("McNemar's statistic is:", mcnemar, " and p value is:", p_value)
    print(conclusion)
        
    # # The critical value approach
    # print("\n--------------------------------------------------------------------------------------")
    # print("Approach 2: The critical value approach to hypothesis testing in the decision rule")
    # critical_value = stats.chi2.ppf(1-sig_level, (len(rows)-1)*(len(columns)-1))
    # conclusion = "Failed to reject the null hypothesis."
    # if mcnemar > critical_value:
    #     conclusion = "Null Hypothesis is rejected."
            
    # print("McNemar's statistic is:", mcnemar, " and critical value is:", critical_value)
    # print(conclusion)
    return {
        "statistic": mcnemar,
        "p_value": p_value,
        "conclusion": conclusion,
    }

def compute_fisher_exact(df, sig_level=0.01):
    # create contingency table
    data_crosstab = pd.crosstab(df['model'],
                                df['human'])
    pprint.pprint(data_crosstab)

    res = stats.fisher_exact(data_crosstab.to_numpy())
    p_value = res.pvalue
    conclusion = "Failed to reject the null hypothesis."

    if p_value <= sig_level:
        conclusion = "Null Hypothesis is rejected."
            
    print("Fisher's exact statistic is:", res.statistic, " and p value is:", p_value)
    print(conclusion)

    return {
        "statistic": res.statistic,
        "p_value": p_value,
        "conclusion": conclusion,
    }

def compute_binomial_test(df, sig_level=0.01):
    # create contingency table
    data_crosstab = pd.crosstab(df['model'],
                                df['human'], margins=True, margins_name="Total")
    pprint.pprint(data_crosstab)
    human_k = data_crosstab.loc["Total", "correct"]
    model_k = data_crosstab.loc["correct", "Total"]
    n = data_crosstab.loc["Total", "Total"]
    p = model_k / n
    print("Human k:", human_k, "Model k:", model_k, "n:", n, "p:", p)
    res = stats.binomtest(k=human_k, n=n, p=p, alternative="greater")
    p_value = res.pvalue
    conclusion = "Failed to reject the null hypothesis."

    if p_value <= sig_level:
        conclusion = "Null Hypothesis is rejected."
            
    print("Binomial exact statistic is:", res.statistic, " and p value is:", p_value)
    print(conclusion)

    return {
        "statistic": res.statistic,
        "p_value": p_value,
        "conclusion": conclusion,
    }

def compute_pitman(df, sig_level=0.01, sample_size=10000):
    model_wins = 0
    human_wins = 0

    for i in tqdm(range(sample_size)):
        model_preds = []
        human_preds = []

        for _, row in df.iterrows():
            coin = random.random()

            if coin > 0.5:
                model_preds.append(1 if row["human"] == "correct" else 0)
                human_preds.append(1 if row["model"] == "correct" else 0)
            else:
                model_preds.append(1 if row["model"] == "correct" else 0)
                human_preds.append(1 if row["human"] == "correct" else 0)
        
        model_accuracy = sum(model_preds) / len(model_preds)
        human_accuracy = sum(human_preds) / len(human_preds)

        if model_accuracy > human_accuracy:
            model_wins += 1
        elif human_accuracy > model_accuracy:
            human_wins += 1
        else:
            pass
    
    statistic = (human_wins - model_wins) / sample_size
    print("Model wins:", model_wins / sample_size, "Human wins:", human_wins / sample_size)
    print("Pitman's test statistic is:", statistic)
    conclusion = "Failed to reject the null hypothesis."

    if statistic > 0:
        conclusion = "Null Hypothesis is rejected."

    return {
        "times": sample_size,
        "model_wins": model_wins / sample_size,
        "human_wins": human_wins / sample_size,
        "statistic": statistic,
        "conclusion": conclusion,
        "p_value": None,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-results", type=str, help="Path to evaluation model results", required=True)
    parser.add_argument("--human-results", type=str, help="Path to evaluation human results", required=True)
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--test", type=str, help="Test name", default="binomial")
    parser.add_argument("--sig-level", type=float, help="Significance level", default=0.05)
    parser.add_argument("--output-dir", type=str, default="stats_test_outputs", help="Output directory")
    
    args = parser.parse_args()

    model_results_path = pathlib.Path(args.model_results)

    model_results = None
    model_results_file_path =model_results_path

    if model_results_path.is_dir():
        json_files = find_json_files(model_results_path)
        model_results_file = [file for file in json_files if args.task in file and "metrics" not in file][0]
        model_results = read_json(model_results_file)
        model_results_file_path = pathlib.Path(model_results_file)
    else:
        model_results = read_json(args.model_results)
    
    human_results = read_json(args.human_results)

    model_results_map = {}

    data = {"model": [], "human": []}
    task_info = TASK_MAP[args.task]

    for result in model_results["data"]:
        model_results_map[result[task_info["model_id_attr"]]] = result

    for human_result in human_results:
        human_id = human_result[task_info["human_id_attr"]]

        if args.task in ["mt_zh_en", "mt_en_de", "mt_en_fr", "mt_en_ru"]:
            human_id = human_result[task_info["human_id_attr"]] + "-" + str(human_result["label"])
        
        if human_id in model_results_map:
            model_result = model_results_map[human_id]
            data["model"].append("correct" if model_result["correct"] else "incorrect")
            data["human"].append("correct" if human_result["correct"] else "incorrect")
        elif args.task == "intent":
            for i in range(10):
                human_id = f'{human_result["data_id"]}-intent-{i}'
                if human_id in model_results_map:
                    model_result = model_results_map[human_id]
                    if human_result["label"] == model_result["intent"]["label"] and human_result["intent"] == model_result["intent"]["text"]:
                        data["model"].append("correct" if model_result["correct"] else "incorrect")
                        data["human"].append("correct" if human_result["correct"] else "incorrect")
                        break
        else:
            # for result in model_results["data"]:
            #     if "model_target_subattr" in task and isinstance(task["model_target_subattr"], list):
            #         for m_tgt_attr, h_tgt_attr in zip(task["model_target_subattr"], task["human_target_attr"]):
            #             if human_result[h_tgt_attr] != result[m_tgt_attr]:
            #                 break

            #     if human_result["label"] == model_result[task["model_target_attr"]]["label"]:
            #         human_result["intent"] == model_result["intent"]["text"]
            print("Human ID {} not found in model results".format(human_id))

    df = pd.DataFrame(data) 

    test_results = {}

    if args.test == "mcnemar":
        test_results = computer_mcnemar(df, args.sig_level)
    elif args.test == "fisher":
        test_results = compute_fisher_exact(df, args.sig_level)
    elif args.test == "binomial":
        test_results = compute_binomial_test(df, args.sig_level)
    elif args.test == "pitman":
        test_results = compute_pitman(df, args.sig_level)
    else:
        raise ValueError("Unknown test type: {}".format(args.test))

    output_dir = pathlib.Path(f"{args.output_dir}/{args.test}")
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "test": args.test,
        "task": args.task,
        "model_path": str(model_results_file_path),
        "human_path": args.human_results,
        "sig_level": args.sig_level,
        "results": test_results
    }

    write_json(outputs, output_dir / f"{model_results_file_path.stem}_stat_test_{args.test}.json")

if __name__ == "__main__":
    main()