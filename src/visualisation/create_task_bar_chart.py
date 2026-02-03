import pandas as pd
from plotnine import (
    element_text,
    element_line,
    coord_flip,
    ggplot,
    aes,
    geom_boxplot,
    scale_fill_manual,
    scale_color_manual,
    facet_wrap,
    labs,
    theme,
    element_rect,
)

EVAL_RESULTS_PATH = "./results"
RESULTS_REFORMATTED_PATH = f"{EVAL_RESULTS_PATH}/vis/bar-chart-reformatted.csv"
VIS_OUTPUT_PATH = "./vis/results-bar-chart.png"

tasks = {
    "token-level-intransitive-nom-subj": "INTRANSITIVE NOMINATIVE (SUBJECT)",
    "token-level-transitive-nom-dat-subj": "NOMINATIVE - DATIVE (SUBJECT)",
    "token-level-transitive-nom-dat-obj": "NOMINATIVE - DATIVE (OBJECT)",
    "token-level-transitive-erg-nom-subj": "ERGATIVE - NOMINATIVE (SUBJECT)",
    "token-level-transitive-erg-nom-obj": "ERGATIVE - NOMINATIVE (OBJECT)",
    "token-level-transitive-dat-nom-subj": "DATIVE - NOMINATIVE (SUBJECT)",
    "token-level-transitive-dat-nom-obj": "DATIVE - NOMINATIVE (OBJECT)",
}

results_reformatted = pd.DataFrame()

for task in tasks.keys():
    results_path_task = f"{EVAL_RESULTS_PATH}/means/{task}.csv"
    task_results_df = pd.read_csv(results_path_task)

    print("====Creating Proportion Chart====")
    print(task_results_df.head())

    long_df = task_results_df.melt(
        id_vars=["model", "repo"], var_name="form", value_name="prob"
    )
    long_df["form"] = long_df["form"].apply(
        lambda x: x.replace("form_grammatical_", "")
        .replace("form_ungrammatical_", "")
        .upper()
    )
    long_df["task"] = tasks[task]

    results_reformatted = pd.concat([results_reformatted, long_df])


results_reformatted.to_csv(RESULTS_REFORMATTED_PATH)

results_reformatted["form"] = pd.Categorical(
    results_reformatted["form"], categories=["NOM", "ERG", "DAT"], ordered=True
)
results_reformatted["task"] = pd.Categorical(
    results_reformatted["task"], categories=tasks.values(), ordered=True
)

colors = {
    "NOM": "#f2710e",
    "ERG": "#5c30b5",
    "DAT": "#0e93f2",
}

print("Results reformatted for bar chart output:")
print(results_reformatted.head())

p = (
    ggplot(results_reformatted, aes(x="form", y="prob", color="form"))
    + geom_boxplot(aes(fill="form"), alpha=0.8, color="black", outlier_shape=None)
    + facet_wrap("~ task", ncol=1)
    + coord_flip()
    + labs(y="Average Model Probability", x="Form", fill="Form")
    + scale_fill_manual(values=colors)
    + scale_color_manual(values=colors)
    + theme(
        panel_background=element_rect(fill="white", color=None),
        strip_background=element_rect(fill="white", color=None),
        plot_background=element_rect(fill="white", color=None),
        # Keep axes
        axis_line=element_line(color="black"),
        axis_ticks=element_line(color="black"),
        # Text clarity
        axis_text=element_text(size=11),
        axis_title=element_text(size=13, weight="bold"),
        strip_text=element_text(size=12, weight="bold"),
        legend_title=element_text(size=12, weight="bold"),
        legend_text=element_text(size=11),
        axis_text_x=element_text(size=12, color="black"),
        axis_text_y=element_text(size=12, color="black"),
    )
)

p.save(VIS_OUTPUT_PATH, width=5, height=15, dpi=300)
