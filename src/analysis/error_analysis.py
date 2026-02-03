from scipy.stats import pearsonr, spearmanr
from scipy.special import logsumexp
from transformers import AutoTokenizer
from transliterate import translit
import pandas as pd
import numpy as np
from pathlib import Path
import os

# All models used in the evaluation and their Hugging Face repos.
models = {
    "HPLT": "HPLT/hplt_bert_base_ka",
    "RemBERT": "google/rembert",
    "mBERT": "google-bert/bert-base-multilingual-cased",
    "XLM-RoBERTa(bs)": "FacebookAI/xlm-roberta-base",
    "XLM-RoBERTa(lg)": "FacebookAI/xlm-roberta-large",
    "mGPT-1.3B-Georgian": "ai-forever/mGPT-1.3B-georgian",
    "GPT2-GEO": "Kuduxaaa/gpt2-geo",
}

# Which datasets test which grammatical case i.e. nominative, dative, or ergative.
config_splits = {
    "NOM": [
        "INTR-S1-NOM-word-level",
        "S1-NOM-word-level",
        "S2-NOM-word-level",
        "S3-NOM-word-level",
    ],
    "DAT": ["S1-DAT-word-level", "S3-DAT-word-level"],
    "ERG": ["S2-ERG-word-level"],
}

# Individual dataset configuration e.g. which cases are grammatical or ungrammatical.
dataset_configs = {
    "INTR-S1-NOM-word-level": {
        "task": "intransitive-nom-subj",
        "grammatical_col": "nom",
        "ungrammatical_cols": ["dat", "erg"],
    },
    "S1-NOM-word-level": {
        "task": "transitive-nom-dat-subj",
        "grammatical_col": "nom",
        "ungrammatical_cols": ["dat", "erg"],
    },
    "S1-DAT-word-level": {
        "task": "transitive-nom-dat-obj",
        "grammatical_col": "dat",
        "ungrammatical_cols": ["nom", "erg"],
    },
    "S2-ERG-word-level": {
        "task": "transitive-erg-nom-subj",
        "grammatical_col": "erg",
        "ungrammatical_cols": ["nom", "dat"],
    },
    "S2-NOM-word-level": {
        "task": "transitive-erg-nom-obj",
        "grammatical_col": "nom",
        "ungrammatical_cols": ["erg", "dat"],
    },
    "S3-DAT-word-level": {
        "task": "transitive-dat-nom-subj",
        "grammatical_col": "dat",
        "ungrammatical_cols": ["nom", "erg"],
    },
    "S3-NOM-word-level": {
        "task": "transitive-dat-nom-obj",
        "grammatical_col": "nom",
        "ungrammatical_cols": ["dat", "erg"],
    },
}


# Computes the number of tokens for a model in the grammatical feature as well as the full sentence.
def num_tokens(data, model: str, grammatical_feature: str):
    tokeniser = AutoTokenizer.from_pretrained(model)

    def tokenise(t: str):
        return tokeniser.tokenize(t)

    data["word_token_length"] = data[f"form_grammatical_{grammatical_feature}"].apply(
        tokenise
    )
    data["word_token_length"] = data["word_token_length"].apply(len)

    data["sent_token_length"] = data["masked_text"].apply(tokenise)
    data["sent_token_length"] = data["sent_token_length"].apply(len)
    return data


def analyse_sentences(data, grammatical_feature: str, evaluation_type, model):
    ungrammatical_features = ["nom", "erg", "dat"]
    ungrammatical_features.remove(grammatical_feature)

    eps = 1e-12  # prevents log(0)

    data = num_tokens(data, model, grammatical_feature)

    if evaluation_type == "word-level":
        logp_g = np.log(data[f"p_form_grammatical_{grammatical_feature}"] + eps)
        logp_ug1 = np.log(
            data[f"p_form_ungrammatical_{ungrammatical_features[0]}"] + eps
        )
        logp_ug2 = np.log(
            data[f"p_form_ungrammatical_{ungrammatical_features[1]}"] + eps
        )
    else:
        logp_g = data[f"I_form_grammatical_{grammatical_feature}"]
        logp_ug1 = data[f"I_form_ungrammatical_{ungrammatical_features[0]}"]
        logp_ug2 = data[f"I_form_ungrammatical_{ungrammatical_features[1]}"]

    data["log_odds"] = logp_g - logsumexp(np.vstack([logp_ug1, logp_ug2]).T, axis=1)
    data[f"log_odds_{grammatical_feature}_{ungrammatical_features[0]}"] = (
        logp_g - logp_ug1
    )
    data[f"log_odds_{grammatical_feature}_{ungrammatical_features[1]}"] = (
        logp_g - logp_ug2
    )

    pref_ug1 = sum(
        data[f"p_form_ungrammatical_{ungrammatical_features[0]}"]
        > data[f"p_form_ungrammatical_{ungrammatical_features[1]}"]
    ) / len(data)

    pears_wl, pears_wl_pval = pearsonr(data["word_token_length"], data["log_odds"])
    pears_wl_char, pears_wl_pval_char = pearsonr(data["word_length"], data["log_odds"])
    spearm_wl_char, spearm_wl_char_pval = spearmanr(
        data["word_token_length"], data["log_odds"]
    )

    pears_wl_erg_nom, _ = pearsonr(data["word_token_length"], data["log_odds_erg_nom"])
    spearm_wl_erg_nom, _ = spearmanr(
        data["word_token_length"], data["log_odds_erg_nom"]
    )

    pears_wl_erg_dat, _ = pearsonr(data["word_token_length"], data["log_odds_erg_dat"])
    spearm_wl_erg_dat, _ = spearmanr(
        data["word_token_length"], data["log_odds_erg_dat"]
    )

    pears_sent, pears_sent_pval = pearsonr(data["sent_token_length"], data["log_odds"])
    spearm_sent, spearm_sent_pval = spearmanr(
        data["sent_token_length"], data["log_odds"]
    )

    data["translit"] = data[f"form_grammatical_{grammatical_feature}"].apply(
        lambda x: translit(x, "ka", reversed=True) if isinstance(x, str) else x
    )

    return data, {
        "pears_wl": pears_wl,
        "pears_wl_pval": pears_wl_pval,
        "pears_wl_char": pears_wl_char,
        "pears_wl_pval_char": pears_wl_pval_char,
        f"pref_{ungrammatical_features[0]}": pref_ug1,
        "spearm_wl_char": spearm_wl_char,
        "spearm_wl_char_pval": spearm_wl_char_pval,
        "pears_wl_erg_nom": pears_wl_erg_nom,
        "spearm_wl_erg_nom": spearm_wl_erg_nom,
        "pears_wl_erg_dat": pears_wl_erg_dat,
        "spearm_wl_erg_dat": spearm_wl_erg_dat,
        "pears_sent": pears_sent,
        "pears_sent_pval": pears_sent_pval,
        "spearm_sent": spearm_sent,
        "spearm_sent_pval": spearm_sent_pval,
    }


def main():
    RESULTS_PATH = "./results/full"

    task_results = {}

    for focus in dataset_configs.keys():
        task_results[focus] = []
        grammatical_feature = dataset_configs[focus]["grammatical_col"]

        matching_files = []
        all_data = []
        for filename in os.listdir(RESULTS_PATH):
            if focus in filename:
                matching_files.append(filename)
        print("Matching files: ", matching_files)

        for filename in matching_files:
            full_path = Path(RESULTS_PATH) / filename
            df = pd.read_csv(full_path)
            all_data.append(df)

        all_token_lengths = {}
        for filename, data in zip(matching_files, all_data):
            for key, value in models.items():
                if key in filename:

                    model = value
                    print("Setting model to ", model)
                    break

            result, correlations = analyse_sentences(
                data, grammatical_feature, "word-level", model
            )

            print("====")
            print("Filename: ", filename)
            all_token_lengths[model] = result["word_token_length"].mean()
            for key, value in correlations.items():
                print(key, round(value, 2))
            print("\n\n\n")
            for model, tl in all_token_lengths.items():
                print(f"{model}: {tl}")
            print("====")
            print("\n\n\n")

            task_results[focus].append(correlations)

    for form, tasks in config_splits.items():
        all_task_results = [task_results[t] for t in tasks]

        all_correlations = []
        for task_result in all_task_results:
            for correlations in task_result:
                print("Adding correlation:")
                print(correlations)
                print("\n\n")
                all_correlations.append(correlations)

        full_df = pd.concat(
            [pd.DataFrame([c]) for c in all_correlations], ignore_index=True
        )
        print(full_df)

        print("Processing...")
        print(full_df.head())
        print("Tasks: ", tasks)
        print("\n\n")

        print(f"===={form}====")
        print("Dataset Size: ", len(full_df))
        print("Results:\n")
        for c in full_df.columns:
            print(f"Avg {c}: ", full_df[c].mean())
            print(f"StdDev {c}: ", full_df[c].std())

        print(
            f"{form} Word: ({full_df["pears_wl"].mean()},p={full_df["pears_wl_pval"].mean()})"
        )
        print(
            f"{form} Sentence: ({full_df["pears_sent"].mean()},p={full_df["pears_sent_pval"].mean()})"
        )
        print("====")
        print("\n\n")


if __name__ == "__main__":
    main()
