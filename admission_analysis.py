import pandas as pd
import os

# -----------------------------------------
# 1. IMPORT DATASET
# -----------------------------------------

file_name = "admission_data.csv"

df = pd.read_csv(file_name)

print("\n========== UNIVERSITY ADMISSION ANALYTICS ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# -----------------------------------------
# 2. DATASET PROFILING
# -----------------------------------------

print("\n========== DATASET PROFILING ==========")

print("\nData Types:")
print(df.dtypes)

print("\nNull Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nMemory Usage:")
print(df.memory_usage(deep=True).sum(), "bytes")

print("\nUnique Values:")
for column in df.columns:
    print(column, ":", df[column].nunique())


# -----------------------------------------
# 3. CLASSIFY ATTRIBUTES
# -----------------------------------------

def classify_attributes(data):

    numerical = []
    categorical = []
    ordinal = []

    for column in data.columns:

        if data[column].dtype in ['int64', 'float64']:
            numerical.append(column)

        elif column == "Board_Percentage":
            ordinal.append(column)

        else:
            categorical.append(column)

    return numerical, categorical, ordinal


numerical, categorical, ordinal = classify_attributes(df)

print("\n========== ATTRIBUTE CLASSIFICATION ==========")

print("\nNumerical Attributes:")
print(numerical)

print("\nCategorical Attributes:")
print(categorical)

print("\nOrdinal Attributes:")
print(ordinal)


# -----------------------------------------
# 4. IDENTIFY INCONSISTENCIES
# -----------------------------------------

print("\n========== DATA INCONSISTENCIES ==========")

# Duplicate application numbers
duplicate_apps = df[df.duplicated("Application_No", keep=False)]

print("\nDuplicate Application Numbers:")

if duplicate_apps.empty:
    print("No duplicate application numbers found.")
else:
    print(duplicate_apps[["Application_No", "Name"]])


# Invalid entrance marks
invalid_entrance = df[
    (df["Entrance_Score"] < 0) |
    (df["Entrance_Score"] > 100)
]

print("\nInvalid Entrance Scores:")

if invalid_entrance.empty:
    print("No invalid entrance scores found.")
else:
    print(invalid_entrance)


# Invalid board percentage
invalid_board = df[
    (df["Board_Percentage"] < 0) |
    (df["Board_Percentage"] > 100)
]

print("\nInvalid Board Percentages:")

if invalid_board.empty:
    print("No invalid board percentages found.")
else:
    print(invalid_board)


# Incorrect age
invalid_age = df[
    (df["Age"] < 15) |
    (df["Age"] > 30)
]

print("\nIncorrect Age Records:")

if invalid_age.empty:
    print("No incorrect age records found.")
else:
    print(invalid_age)


# -----------------------------------------
# 5. ADMISSION STATISTICS REPORT
# -----------------------------------------

print("\n========== ADMISSION STATISTICS ==========")

total_students = len(df)

admitted = (df["Admission_Status"] == "Admitted").sum()

not_admitted = (df["Admission_Status"] == "Not Admitted").sum()

admission_percentage = (admitted / total_students) * 100

print("\nTotal Applications:", total_students)

print("Admitted Students:", admitted)

print("Not Admitted Students:", not_admitted)

print("Admission Percentage:",
      round(admission_percentage, 2), "%")


print("\nAverage Entrance Score:",
      round(df["Entrance_Score"].mean(), 2))

print("Average Board Percentage:",
      round(df["Board_Percentage"].mean(), 2))

print("Average Family Income:",
      round(df["Family_Income"].mean(), 2))


print("\nAdmissions by Branch:")
print(df.groupby("Branch")["Admission_Status"]
      .apply(lambda x: (x == "Admitted").sum()))


print("\nAdmissions by Category:")
print(df.groupby("Category")["Admission_Status"]
      .apply(lambda x: (x == "Admitted").sum()))


# -----------------------------------------
# 6. CLEAN DATASET
# -----------------------------------------

print("\n========== CLEANING DATA ==========")

# Remove duplicate application numbers
df_cleaned = df.drop_duplicates(
    subset=["Application_No"],
    keep="first"
)

# Remove invalid entrance scores
df_cleaned = df_cleaned[
    (df_cleaned["Entrance_Score"] >= 0) &
    (df_cleaned["Entrance_Score"] <= 100)
]

# Remove invalid board percentages
df_cleaned = df_cleaned[
    (df_cleaned["Board_Percentage"] >= 0) &
    (df_cleaned["Board_Percentage"] <= 100)
]

# Remove incorrect ages
df_cleaned = df_cleaned[
    (df_cleaned["Age"] >= 15) &
    (df_cleaned["Age"] <= 30)
]

print("\nOriginal Records:", len(df))

print("Cleaned Records:", len(df_cleaned))


# -----------------------------------------
# 7. EXPORT CLEANED DATASET
# -----------------------------------------

output_file = "cleaned_admission_data.csv"

df_cleaned.to_csv(output_file, index=False)

print("\nCleaned dataset exported successfully!")

print("File Name:", output_file)

print("\n========== PROGRAM COMPLETED ==========")