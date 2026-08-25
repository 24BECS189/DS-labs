import pandas as pd
import numpy as np

print("\n========== EXPERIMENT 3 ==========")
print("ADVANCED DATA CLEANING AND FEATURE ENGINEERING")


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("customer_data.csv")

print("\n========== ORIGINAL DATASET ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Records:")
print(df.head())


# =====================================================
# 2. DETECT DATA QUALITY ISSUES
# =====================================================

print("\n========== DATA QUALITY ISSUES ==========")

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Duplicate customer IDs
print("\nDuplicate Customer IDs:")

duplicates = df[df.duplicated("Customer_ID", keep=False)]

if duplicates.empty:
    print("No duplicates found")
else:
    print(duplicates[["Customer_ID", "Name"]])


# Incorrect ages
print("\nIncorrect Age Values:")

invalid_age = df[
    (df["Age"] < 18) |
    (df["Age"] > 100)
]

print(invalid_age[["Customer_ID", "Age"]])


# Currency mismatch
print("\nCurrency Values:")
print(df["Currency"].value_counts())


# Date formats
print("\nDate Values:")
print(df["Join_Date"].head(10))


# Outliers using IQR
Q1 = df["Total_Spending"].quantile(0.25)
Q3 = df["Total_Spending"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["Total_Spending"] < lower_limit) |
    (df["Total_Spending"] > upper_limit)
]

print("\nSpending Outliers:")
print(
    outliers[
        ["Customer_ID", "Total_Spending"]
    ]
)


# =====================================================
# 3. IMPUTATION TECHNIQUES
# =====================================================

print("\n========== IMPUTATION COMPARISON ==========")

# Mean
mean_df = df.copy()

mean_value = mean_df["Monthly_Salary"].mean()

mean_df["Monthly_Salary"] = mean_df[
    "Monthly_Salary"
].fillna(mean_value)

print("\nMean Imputation:")
print("Missing Salary:", mean_df["Monthly_Salary"].isnull().sum())
print("Filled Value:", mean_value)


# Median
median_df = df.copy()

median_value = median_df["Monthly_Salary"].median()

median_df["Monthly_Salary"] = median_df[
    "Monthly_Salary"
].fillna(median_value)

print("\nMedian Imputation:")
print("Missing Salary:", median_df["Monthly_Salary"].isnull().sum())
print("Filled Value:", median_value)


# Mode
mode_df = df.copy()

mode_value = mode_df["Monthly_Salary"].mode()[0]

mode_df["Monthly_Salary"] = mode_df[
    "Monthly_Salary"
].fillna(mode_value)

print("\nMode Imputation:")
print("Missing Salary:", mode_df["Monthly_Salary"].isnull().sum())
print("Filled Value:", mode_value)


# Forward Fill
ffill_df = df.copy()

ffill_df["Monthly_Salary"] = ffill_df[
    "Monthly_Salary"
].ffill()

print("\nForward Fill:")
print("Missing Salary:", ffill_df["Monthly_Salary"].isnull().sum())


# Backward Fill
bfill_df = df.copy()

bfill_df["Monthly_Salary"] = bfill_df[
    "Monthly_Salary"
].bfill()

print("\nBackward Fill:")
print("Missing Salary:", bfill_df["Monthly_Salary"].isnull().sum())


# =====================================================
# 4. CLEAN DATASET
# =====================================================

print("\n========== DATA CLEANING ==========")

cleaned = df.copy()


# Fill missing salary using median
cleaned["Monthly_Salary"] = cleaned[
    "Monthly_Salary"
].fillna(
    cleaned["Monthly_Salary"].median()
)


# Remove duplicate customer IDs
cleaned = cleaned.drop_duplicates(
    subset=["Customer_ID"],
    keep="first"
)


# Fix invalid age values
median_age = cleaned[
    (cleaned["Age"] >= 18) &
    (cleaned["Age"] <= 100)
]["Age"].median()

cleaned.loc[
    (cleaned["Age"] < 18) |
    (cleaned["Age"] > 100),
    "Age"
] = median_age


# Convert USD to INR
cleaned.loc[
    cleaned["Currency"] == "USD",
    "Monthly_Salary"
] = cleaned.loc[
    cleaned["Currency"] == "USD",
    "Monthly_Salary"
] * 83

cleaned.loc[
    cleaned["Currency"] == "USD",
    "Total_Spending"
] = cleaned.loc[
    cleaned["Currency"] == "USD",
    "Total_Spending"
] * 83

cleaned["Currency"] = "INR"


# Convert dates into one format
cleaned["Join_Date"] = pd.to_datetime(
    cleaned["Join_Date"],
    dayfirst=True,
    errors="coerce"
)

cleaned["Join_Date"] = cleaned[
    "Join_Date"
].dt.strftime("%Y-%m-%d")


print("Missing salary fixed")
print("Duplicate IDs removed")
print("Invalid ages corrected")
print("Currency converted to INR")
print("Dates standardized")


# =====================================================
# 5. STANDARDIZATION
# =====================================================

print("\n========== STANDARDIZATION ==========")

salary_mean = cleaned["Monthly_Salary"].mean()
salary_std = cleaned["Monthly_Salary"].std()

cleaned["Salary_Standardized"] = (
    cleaned["Monthly_Salary"] - salary_mean
) / salary_std

print(
    cleaned[
        ["Customer_ID", "Salary_Standardized"]
    ].head()
)


# =====================================================
# 6. NORMALIZATION
# =====================================================

print("\n========== NORMALIZATION ==========")

min_spending = cleaned["Total_Spending"].min()
max_spending = cleaned["Total_Spending"].max()

cleaned["Spending_Normalized"] = (
    cleaned["Total_Spending"] - min_spending
) / (max_spending - min_spending)

print(
    cleaned[
        ["Customer_ID", "Spending_Normalized"]
    ].head()
)


# =====================================================
# 7. LABEL ENCODING
# =====================================================

print("\n========== LABEL ENCODING ==========")

gender_mapping = {
    "Female": 0,
    "Male": 1
}

cleaned["Gender_Label"] = cleaned[
    "Gender"
].map(gender_mapping)

print(
    cleaned[
        ["Gender", "Gender_Label"]
    ].head()
)


# =====================================================
# 8. ONE-HOT ENCODING
# =====================================================

print("\n========== ONE-HOT ENCODING ==========")

cleaned = pd.get_dummies(
    cleaned,
    columns=["City"],
    dtype=int
)

print(cleaned.head())


# =====================================================
# 9. FEATURE ENGINEERING
# =====================================================

print("\n========== FEATURE ENGINEERING ==========")


# Annual Income
cleaned["Annual_Income"] = (
    cleaned["Monthly_Salary"] * 12
)


# Age Group
def age_group(age):

    if age < 25:
        return "Young"

    elif age < 40:
        return "Adult"

    else:
        return "Senior"


cleaned["Age_Group"] = cleaned[
    "Age"
].apply(age_group)


# Spending Category
def spending_category(spending):

    if spending < 150000:
        return "Low"

    elif spending < 300000:
        return "Medium"

    else:
        return "High"


cleaned["Spending_Category"] = cleaned[
    "Total_Spending"
].apply(spending_category)


# Customer Value Index
cleaned["Customer_Value_Index"] = (
    cleaned["Annual_Income"] / 1000
    + cleaned["Total_Spending"] / 10000
)


print(
    cleaned[
        [
            "Customer_ID",
            "Annual_Income",
            "Age_Group",
            "Spending_Category",
            "Customer_Value_Index"
        ]
    ]
)


# =====================================================
# 10. BEFORE AND AFTER COMPARISON
# =====================================================

print("\n========== BEFORE VS AFTER ==========")

print("\nBefore Preprocessing:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing Values:", df.isnull().sum().sum())
print("Duplicate IDs:", df["Customer_ID"].duplicated().sum())


print("\nAfter Preprocessing:")
print("Rows:", cleaned.shape[0])
print("Columns:", cleaned.shape[1])
print("Missing Values:", cleaned.isnull().sum().sum())
print("Duplicate IDs:", cleaned["Customer_ID"].duplicated().sum())


# =====================================================
# 11. ANALYTICAL OBSERVATIONS
# =====================================================

print("\n========== ANALYTICAL OBSERVATIONS ==========")

print(
    "1. Missing salary values were handled using median imputation."
)

print(
    "2. Duplicate customer IDs were removed."
)

print(
    "3. Invalid age values were corrected."
)

print(
    "4. Currency values were standardized to INR."
)

print(
    "5. Date formats were converted into a common format."
)

print(
    "6. Standardization and normalization improved numerical consistency."
)

print(
    "7. New customer features were created for better analysis."
)


# =====================================================
# 12. SAVE DATASET
# =====================================================

output_file = "processed_customer_data.csv"

cleaned.to_csv(
    output_file,
    index=False
)

print("\n========== DATA SAVING ==========")

print(
    "Processed dataset saved successfully!"
)

print("File Name:", output_file)

print("\n========== PROGRAM COMPLETED ==========")