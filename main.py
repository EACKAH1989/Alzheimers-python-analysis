import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

df = pd.read_csv("/Users/ataka/Downloads/alzheimers_disease_data.csv")

print(df.head())
# Drop column

df = df.drop(columns=["DoctorInCharge"])

# Clean column names (like janitor::clean_names)

df.columns = (

    df.columns

    .str.strip()

    .str.lower()

    .str.replace(" ", "_")

)

# Recode categorical variables

df["gender"] = df["gender"].map({0: "Male", 1: "Female"})

df["diagnosis"] = df["diagnosis"].map({0: "No", 1: "Yes"})

df["smoking"] = df["smoking"].map({0: "No", 1: "Yes"})

df["diabetes"] = df["diabetes"].map({0: "No", 1: "Yes"})

df["hypertension"] = df["hypertension"].map({0: "No", 1: "Yes"})

df["depression"] = df["depression"].map({0: "No", 1: "Yes"})

# Convert to categorical (optional but cleaner)

cat_cols = ["gender", "diagnosis", "smoking", "diabetes", "hypertension", "depression"]

df[cat_cols] = df[cat_cols].astype("category")

# Create age group (equivalent to cut())

df["age_group"] = pd.cut(

    df["age"],

    bins=[0, 60, 70, 80, 90, 100],

    labels=["<60", "60–69", "70–79", "80–89", "90+"],

    right=False

)

print(df.head(10))
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.lineplot(data=df, x="age", y="mmse", hue="diagnosis")
plt.title("MMSE decline pattern by diagnosis")
plt.savefig("Figures/MMSE decline pattern by diagnosis.png", dpi=300, bbox_inches="tight")
#plt.show()


# Encode categorical variables
df_encoded = df.copy()
df_encoded["diagnosis"] = df_encoded["diagnosis"].map({"No": 0, "Yes": 1})

# Split X and y BEFORE encoding
y = df_encoded["diagnosis"]
X = df_encoded.drop(columns=["diagnosis"])

# One-hot encode predictors
X = pd.get_dummies(X, drop_first=True)

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=42, stratify=y

)

#Train model
model = RandomForestClassifier(

    n_estimators=300,

    max_depth=5,

    random_state=42

)

model.fit(X_train, y_train)

#Evaluate model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")
print(classification_report(y_test, y_pred))

#Permutation Importance
perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importances = pd.Series(perm.importances_mean, index=X.columns)

#Plot Top Features
top_features = importances.sort_values().tail(10)
plt.figure()
top_features.plot(kind="barh")
plt.title("Top 10 Feature Importance (Permutation)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("Figures/feature_importance.png", dpi=300, bbox_inches="tight")
#plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "figure.dpi": 300
})

# Color-blind friendly palette
COLORS = {
    "No": "#4C72B0",   # muted blue
    "Yes": "#DD8452"   # muted orange
}
import os
os.makedirs("Figures", exist_ok=True)

#AGE vs DIAGNOSIS (BOXPLOT)
plt.figure()
sns.boxplot(
    data=df,
    x="diagnosis",
    y="age",
    hue="diagnosis",
    palette={"No": "#3fb68b", "Yes": "#e05c5c"},
    legend=False,
    width=0.5
)
plt.title("Age Distribution by Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Age")
plt.savefig("Figures/age_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

#MMSE vs DIAGNOSIS (BOXPLOT)
plt.figure(figsize=(6, 4))
sns.boxplot(
    data=df,
    x="diagnosis",
    y="mmse",
    hue="diagnosis",
    palette=COLORS,
    width=0.5,
    fliersize=0,
    linewidth=1,
    legend=False

)
sns.stripplot(
    data=df,
    x="diagnosis",
    y="mmse",
    color="black",
    alpha=0.25,
    size=3
)
plt.title("MMSE Scores by Alzheimer’s Diagnosis", pad=10)
plt.xlabel("")
plt.ylabel("MMSE Score")
sns.despine()
plt.tight_layout()
plt.savefig("Figures/Figure1_MMSE.png")
plt.show()

#PREVLAENCE BY AGE GROUP
prev_df = df.groupby("age_group")["diagnosis"].apply(lambda x: (x == "Yes").mean()).reset_index()
plt.figure(figsize=(6, 4))
sns.barplot(
    data=prev_df,
    x="age_group",
    y="diagnosis",
    color="#4C72B0"
)

# Confidence intervals removed for clean journal style
plt.ylabel("Prevalence")
plt.xlabel("Age Group")
plt.title("Prevalence of Alzheimer’s Disease by Age Group", pad=10)

# Add % labels
for i, val in enumerate(prev_df["diagnosis"]):
    plt.text(i, val + 0.01, f"{val*100:.1f}%", ha="center", fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig("Figures/Figure3_Prevalence.png")
plt.show()

#ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(frameon=False)

sns.despine()
plt.tight_layout()
plt.savefig("Figures/Figure4_ROC.png")
plt.show()

#Feature Importance
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(6, 4))

sns.barplot(
    x=top_features.values,
    y=top_features.index,
    color="#4C72B0"
)

plt.xlabel("Permutation Importance")
plt.ylabel("")
plt.title("Top Predictors of Alzheimer’s Diagnosis", pad=10)

sns.despine()
plt.tight_layout()
plt.savefig("Figures/Figure5_Features.png")
plt.show()