import pandas as pd

df = pd.read_csv("updated_results.csv")

task_order = [
    "INTR-S1-NOM",
    "S1-NOM",
    "S1-DAT",
    "S2-ERG",
    "S2-NOM",
    "S3-DAT",
    "S3-NOM",
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

token_table = make_pivot(token_df)
sent_table  = make_pivot(sent_df)

def accuracy_to_cell(val):
    if pd.isna(val):
        return ""
    intensity = int(val)   
    return rf"\cellcolor{{softgreen!{intensity}}} {val}"

token_latex_df = token_table.applymap(accuracy_to_cell)
sent_latex_df  = sent_table.applymap(accuracy_to_cell)

combined = pd.concat([token_latex_df, sent_latex_df])
combined = combined.to_latex(
    escape=False,
    multicolumn=True,
    multicolumn_format="c",
    header=True
)

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
# print(combined)

