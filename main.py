import pandas as pd


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
plt.show()

