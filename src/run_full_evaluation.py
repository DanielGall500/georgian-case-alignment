from grewtse.evaluators.evaluator import GrewTSEvaluator
from datasets import load_dataset
import pandas as pd
import os

OVERVIEW_OUTPUT_DIR = "./evaluation_results"
OVERVIEW_OUTPUT_FILENAME = "overview"
OVERVIEW_OUTPUT_FULL_PATH = f"{OVERVIEW_OUTPUT_DIR}/{OVERVIEW_OUTPUT_FILENAME}.csv"
DATASET_REPO = "DanielGallagherIRE/georgian-case-alignment"

dataset_config = {
    "INTR-S1-NOM": {
        "task": "intransitive-nom-subj",
        "grammatical_col": "form_grammatical_nom",
        "ungrammatical_cols": [
            "form_ungrammatical_dat",
            "form_ungrammatical_erg"
        ]
    },
    "S1-NOM": {
        "task": "transitive-nom-dat-subj",
        "grammatical_col": "form_grammatical_nom",
        "ungrammatical_cols": [
            "form_ungrammatical_dat",
            "form_ungrammatical_erg"
        ]
    },
    "S1-DAT": {
        "task": "transitive-nom-dat-obj",
        "grammatical_col": "form_grammatical_dat",
        "ungrammatical_cols": [
            "form_ungrammatical_nom",
            "form_ungrammatical_erg"
        ]
    },
    "S2-ERG": {
        "task": "transitive-erg-nom-subj",
        "grammatical_col": "form_grammatical_erg",
        "ungrammatical_cols": [
            "form_ungrammatical_nom",
            "form_ungrammatical_dat"
        ]
    },
    "S2-NOM": {
        "task": "transitive-erg-nom-obj",
        "grammatical_col": "form_grammatical_nom",
        "ungrammatical_cols": [
            "form_ungrammatical_erg",
            "form_ungrammatical_dat"
        ]
    },
    "S3-DAT": {
        "task": "transitive-dat-nom-subj",
        "grammatical_col": "form_grammatical_dat",
        "ungrammatical_cols": [
            "form_ungrammatical_nom",
            "form_ungrammatical_erg"
        ]
    },
    "S3-NOM": {
        "task": "transitive-dat-nom-obj",
        "grammatical_col": "form_grammatical_nom",
        "ungrammatical_cols": [
            "form_ungrammatical_dat",
            "form_ungrammatical_erg"
        ]
    },
}

evaluation_config = {
    "mlm": {
        "mBERT": "google-bert/bert-base-multilingual-cased", 
        "RemBERT": "google/rembert",
        "HPLT": "HPLT/hplt_bert_base_ka",
        "XLM-RoBERTa(bs)": "FacebookAI/xlm-roberta-base",
        "XLM-RoBERTa(lg)": "FacebookAI/xlm-roberta-large",
    },
    "ntp": {
        "mGPT": "ai-forever/mGPT-1.3B-georgian",
        "mGPT-13B": "ai-forever/mGPT-13B",
        "mGPT-1.3B-Georgian": "ai-forever/mGPT-1.3B-georgian",
        "GPT2-GEO": "Kuduxaaa/gpt2-geo",
        "Kona2-12B": "tbilisi-ai-lab/kona2-12B"
    }
}

def main():
    if not os.path.exists(OVERVIEW_OUTPUT_DIR):
        os.makedirs(OVERVIEW_OUTPUT_DIR)
    for task_type, models in evaluation_config.items():
        for model, model_repo in models.items():
            for dataset_name, dataset in dataset_config.items():

                geval = GrewTSEvaluator()

                # eval config setup
                task = dataset["task"]

                # setup the cols that correspond to the grammatical and ungrammtical forms
                grammatical_col = dataset["grammatical_col"]
                ungrammatical_cols = dataset["ungrammatical_cols"]
                evaluation_cols = [grammatical_col]
                evaluation_cols.extend(ungrammatical_cols)

                task_dataset = load_dataset(DATASET_REPO, task)["train"].to_pandas()
                print(task_dataset.head())

                if task_type == "mlm":
                    evaluation_types = ["token-level"]
                elif task_type == "ntp":
                    evaluation_types = ["token-level", "sentence-level"]
                else:
                    raise ValueError("Invalid task type.")

                for eval_type in evaluation_types:
                    evaluation_results = geval.evaluate_model(
                        mp_dataset=task_dataset,
                        model_repo=model_repo,
                        task_type=task_type,
                        evaluation_type=eval_type,
                        evaluation_cols=evaluation_cols,
                        save_to=f"evaluation_results/{task_type}-{model}-{dataset_name}-{eval_type}.csv",
                        device="cuda",
                    )

                    all_probs_grammatical = evaluation_results[f"p_{grammatical_col}"]
                    all_probs_ungrammatical = {col:evaluation_results[f"p_{col}"] for col in ungrammatical_cols}

                    print("====")
                    print(task_type)
                    print(eval_type)
                    print(model)
                    print(task)
                    print("====")
                    print("Avg Prob Grammatical: ", all_probs_grammatical.mean())
                    for ug in all_probs_ungrammatical.keys():
                        print(f"Avg Prob Ungrammatical {ug}: ", all_probs_ungrammatical[ug].mean())
                            

                    accuracy = geval.get_accuracy(
                        f"p_{grammatical_col}", [f"p_{ug_col}" for ug_col in ungrammatical_cols]
                    )
                    print(accuracy)
                    print("=====")

                    if os.path.exists(OVERVIEW_OUTPUT_FULL_PATH):
                        results_df = pd.read_csv(OVERVIEW_OUTPUT_FULL_PATH)
                        results_df.loc[len(results_df)] = [
                                model_repo, task, eval_type, task_type, accuracy
                        ]
                    else:
                        results_df = pd.DataFrame([
                            [model_repo,task,eval_type,task_type,accuracy]
                        ], columns=["model", "task", "evaluation_type", "task_type", "accuracy"])
                    results_df.to_csv(OVERVIEW_OUTPUT_FULL_PATH, index=False)

if __name__ == "__main__":
    main()

