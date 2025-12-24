import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, scale_fill_gradient,
    theme, element_text, labs, facet_wrap, geom_text
)

# Load the data
df = pd.read_csv("updated_results.csv")

# Preserve the order of Tasks as they appear in the CSV
task_order = df["task"].drop_duplicates().tolist()
df["task"] = pd.Categorical(df["task"], categories=task_order, ordered=True)

# Optional: preserve order of Models too
model_order = df["model"].drop_duplicates().tolist()
df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

# Heatmap with green gradient
p = (
    ggplot(df, aes(x="task", y="model", fill="accuracy"))
    + geom_tile(color="white")
    + geom_text(aes(label="accuracy", size=8)) # show accuracy
    + scale_fill_gradient(low="lightgreen", high="darkgreen", limits=(0, 0.8))
    + facet_wrap("~evaluation_type")
    + labs(x="task", y="model", fill="accuracy",
           title="Georgian Case Alignment – TSE Accuracy")
    + theme(
        figure_size=(12, 6),
        axis_text_x=element_text(rotation=10, ha="right"),
        axis_text_y=element_text(size=9),
        title=element_text(size=14, weight="bold")
    )
)

# Save the plot
p.save("updated_georgian_case_heatmap_green.png", dpi=300)

