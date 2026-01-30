import pandas as pd

EVAL_RESULTS_PATH = "./results/full"
RESULTS_REFORMATTED_PATH = f"{EVAL_RESULTS_PATH}/vis/bar-chart-reformatted.csv"
VIS_OUTPUT_PATH = "./vis/results-bar-chart.png"

OVERVIEW_RESULTS_PATH = f"{EVAL_RESULTS_PATH}/overview.csv"

df = pd.read_csv(OVERVIEW_RESULTS_PATH)

print("====Creating Latex Table For Results====")
print(df.head())

task_order = [
    "intransitive-nom-subj",
    "transitive-nom-dat-subj",
    "transitive-nom-dat-obj",
    "transitive-erg-nom-subj",
    "transitive-erg-nom-obj",
    "transitive-dat-nom-subj",
    "transitive-dat-nom-obj",
]

df["task"] = pd.Categorical(
        df["task"],
        categories=task_order,
        ordered=True
)
print(df["accuracy"])
df["accuracy"] = df["accuracy"].apply(lambda x: float(x) * 100)
df["accuracy"] = df["accuracy"].astype(int)

token_df = df[df["evaluation_type"] == "token-level"]
sent_df  = df[df["evaluation_type"] == "sentence-level"]

def make_pivot(df):
    return df.pivot(
        index="model",
        columns="task",
        values="accuracy"
    )

print(token_df)
print("====")
print(sent_df)

# sent_df = sent_df[sent_df["model"] != "ai-forever/mGPT-1.3B-georgian"]
# sent_df = sent_df[sent_df["model"] != "Kuduxaaa/gpt2-geo"]

print(sent_df["model"].unique())

print("Token-level Acc Avg: ", token_df["accuracy"].mean())
print("Sentence-level Acc Avg: ", sent_df["accuracy"].mean())
print("Difference: ", sent_df["accuracy"].mean() - token_df["accuracy"].mean())

token_table = make_pivot(token_df)
sent_table  = make_pivot(sent_df)

def accuracy_to_cell(val):
    if pd.isna(val):
        return ""
    intensity = int(val) -10 
    return rf"\cellcolor{{softgreen!{intensity}}} {val}"

token_latex_df = token_table.applymap(accuracy_to_cell)
sent_latex_df  = sent_table.applymap(accuracy_to_cell)

"""
combined = pd.concat([token_latex_df, sent_latex_df])
combined = combined.to_latex(
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    header=True
)
"""
token_latex = token_latex_df.to_latex(
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    caption="Token-level accuracy by model and task",
    label="tab:token_level",
    header=False
)

sent_latex = sent_latex_df.to_latex(
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    caption="Sentence-level accuracy by model and task",
    label="tab:sentence_level",
    header=False
)

print(token_latex)
print("\n\n")
print(sent_latex)

