import pandas as pd

RESULTS_PATH = "./results/overview.csv"
REFORMATTED_RESULTS_PATH = "./results/vis/bar-chart-reformatted.csv"
results_df = pd.read_csv(RESULTS_PATH)

# -- Calculate word- and sentence-level average accuracies by case--
dataset_split_by_case = {
    "NOM": [
        "intransitive-nom-subj",
        "transitive-nom-dat-subj",
        "transitive-erg-nom-obj",
        "transitive-dat-nom-obj",
    ],
    "DAT": [
        "transitive-nom-dat-obj",
        "transitive-dat-nom-subj",
    ],
    "ERG": [
        "transitive-erg-nom-subj",
    ],
}

metrics = ["word-level", "sentence-level"]

for metric in metrics:
    for case, task in dataset_split_by_case.items():
        df_subset = results_df[
            (results_df["task"].isin(task)) & (results_df["evaluation_type"] == metric)
        ]
        average_accuracy = df_subset["accuracy"].mean()
        average_accuracy_pct = round(average_accuracy * 100, 1)

        print(f"====Analysing: {case}, Metric: {metric}====")
        print("Tasks:", task)
        print(df_subset.head())
        print("Average Accuracy: ", average_accuracy_pct)
        print("========")
        print("\n\n")

# -- Calculate p(x) from reformatted dataset --
# This is computed directly from the calculations used
# for the bar chart showing p(x) averaged by task, so
# we must adjust our dataset split to the labels used
# for that.
dataset_split_by_case_alt = {
    "NOM": [
        "INTRANSITIVE NOMINATIVE (SUBJECT)",
        "NOMINATIVE - DATIVE (SUBJECT)",
        "ERGATIVE - NOMINATIVE (OBJECT)",
        "DATIVE - NOMINATIVE (OBJECT)",
    ],
    "DAT": [
        "NOMINATIVE - DATIVE (OBJECT)",
        "DATIVE - NOMINATIVE (SUBJECT)",
    ],
    "ERG": ["ERGATIVE - NOMINATIVE (SUBJECT)"],
}

reformatted_results = pd.read_csv(REFORMATTED_RESULTS_PATH)
for case, task in dataset_split_by_case_alt.items():
    df_subset = reformatted_results[
        (reformatted_results["task"].isin(task))
    ].reset_index()

    average_probability = round(df_subset["prob"].mean(), 3)

    print(f"====Analysing: {case}, Metric: Word-Level ====")
    print("Tasks:", task)
    print("Found all Tasks: ", any([t in set(df_subset["task"]) for t in task]))
    print(df_subset.head())
    print("Average P(X): ", average_probability)
    print("========")
    print("\n\n")
