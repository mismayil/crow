import argparse
import subprocess

DATA_PATHS = [
    ("dialogue", "dialogue/data/full_data/dialogue_eval_llm_{template}.json"),
    ("intent", "intent/data/full_data/intent_eval_llm_{template}.json"),
    ("safety", "safety/data/full_data/safety_eval_llm_{template}.json"),
    ("stance", "stance/data/full_data/stance_eval_llm_{template}.json"),
    ("summarization", "summarization/data/full_data/summarization_eval_llm_{template}.json"),
    ("mt_en_de", "translation/data/en-de/full_data/mt_en_de_eval_llm_{template}.json"),
    ("mt_en_fr", "translation/data/en-fr/full_data/mt_en_fr_eval_llm_{template}.json"),
    ("mt_en_ru", "translation/data/en-ru/full_data/mt_en_ru_eval_llm_{template}.json"),
    ("mt_zh_en", "translation/data/zh-en/full_data/mt_zh_en_eval_llm_{template}.json")
]
    
def get_script_name(model):
    if model in ["random", "majority"]:
        return "./evaluate_baseline.py"
    
    if model in ["gpt-4", "text-davinci-003"]:
        return "./evaluate_gpt.py"
    
    return "./evaluate_hf.py"

def run_python_script(script_name, arguments):
    subprocess.run(["python", script_name] + arguments)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--template", type=str, default="bcq", help="Dataset template to use")
    parser.add_argument("--model", type=str, default="random", help="Model to use for evaluation")
    parser.add_argument("--run-name", type=str, default="run1", help="Run name for outputs")
    parser.add_argument("--tasks", type=str, nargs="+", default=["dialogue", "intent", "safety", "stance", "summarization", "mt_en_de", "mt_en_fr", "mt_en_ru", "mt_zh_en"], help="Tasks to evaluate on")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="Other script arguments") # invoke with -- in the beginning

    args = parser.parse_args()

    script_name = get_script_name(args.model)

    output_dir = f"outputs/{args.model}/{args.run_name}"

    for task, datapath in DATA_PATHS:
        if task in args.tasks:
            datapath = datapath.format(template=args.template)
            print(f"Running {script_name} for {datapath}")
            base_args = ["--datapath", datapath, "--model", args.model, "--output-dir", output_dir]
            run_python_script(script_name, base_args + args.rest[1:]) # first element is --
    
    # Report metrics
    print("Reporting metrics")
    run_python_script("./report_metrics.py", ["--results-path", output_dir])
    run_python_script("./report_agg_metrics_across_tasks.py", ["--model", args.model, "--results-dir", output_dir])

    print("Done")

if __name__ == "__main__":
    main()