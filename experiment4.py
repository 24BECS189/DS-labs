import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n========== NATIONAL HEALTH MONITORING SYSTEM ==========")


# =====================================================
# 1. LOAD DATASET
# =====================================================

df = pd.read_csv("health_data.csv")

print("\n========== DATASET ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 Records:")
print(df.head())


# =====================================================
# 2. DESCRIPTIVE STATISTICS
# =====================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

medical_columns = [
    "Age",
    "Heart_Rate",
    "Blood_Pressure",
    "Sugar_Level",
    "BMI",
    "Cholesterol"
]

print(
    df[medical_columns].describe()
)


# =====================================================
# 3. ADDITIONAL STATISTICS
# =====================================================

print("\n========== ADDITIONAL STATISTICS ==========")

for column in medical_columns:

    print("\n", column)

    print("Mean:",
          round(df[column].mean(), 2))

    print("Median:",
          round(df[column].median(), 2))

    print("Minimum:",
          df[column].min())

    print("Maximum:",
          df[column].max())

    print("Standard Deviation:",
          round(df[column].std(), 2))


# =====================================================
# 4. HOSPITAL PATIENT DISTRIBUTION
# =====================================================

print("\n========== HOSPITAL PATIENT DISTRIBUTION ==========")

hospital_counts = df["Hospital"].value_counts()

print(hospital_counts)


# Calculate average patients per hospital

average_patients = hospital_counts.mean()

print(
    "\nAverage patients per hospital:",
    round(average_patients, 2)
)

print("\nHospitals with abnormal patient distribution:")

for hospital, count in hospital_counts.items():

    if count > average_patients * 1.5:
        print(
            hospital,
            "has unusually high patient count:",
            count
        )

    elif count < average_patients * 0.5:
        print(
            hospital,
            "has unusually low patient count:",
            count
        )

    else:
        print(
            hospital,
            "has normal patient distribution:",
            count
        )


# =====================================================
# 5. OUTLIERS USING IQR
# =====================================================

print("\n========== IQR OUTLIER DETECTION ==========")

iqr_outliers = {}

for column in medical_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    iqr_outliers[column] = len(outliers)

    print(
        column,
        "->",
        len(outliers),
        "outliers"
    )


# =====================================================
# 6. OUTLIERS USING Z-SCORE
# =====================================================

print("\n========== Z-SCORE OUTLIER DETECTION ==========")

zscore_outliers = {}

for column in medical_columns:

    mean = df[column].mean()

    std = df[column].std()

    z_scores = (
        (df[column] - mean) / std
    )

    outliers = df[
        abs(z_scores) > 3
    ]

    zscore_outliers[column] = len(outliers)

    print(
        column,
        "->",
        len(outliers),
        "outliers"
    )


# =====================================================
# 7. MEAN VS MEDIAN
# =====================================================

print("\n========== MEAN VS MEDIAN ==========")

for column in medical_columns:

    mean_value = df[column].mean()

    median_value = df[column].median()

    print(
        column,
        ": Mean =",
        round(mean_value, 2),
        ", Median =",
        round(median_value, 2)
    )


# =====================================================
# 8. STATISTICAL SUMMARY REPORT
# =====================================================

print("\n========== STATISTICAL SUMMARY REPORT ==========")

summary = df[medical_columns].describe().T

summary["Median"] = df[
    medical_columns
].median()

summary["IQR_Outliers"] = [
    iqr_outliers[column]
    for column in medical_columns
]

summary["ZScore_Outliers"] = [
    zscore_outliers[column]
    for column in medical_columns
]

print(summary)


# Save report

summary.to_csv(
    "health_statistical_report.csv"
)

print(
    "\nStatistical report saved as:",
    "health_statistical_report.csv"
)


# =====================================================
# 9. VISUALIZATION
# =====================================================

print("\n========== VISUALIZATION ==========")

# Histogram of Blood Sugar

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sugar_Level"],
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of Sugar Level")

plt.xlabel("Sugar Level")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("sugar_distribution.png")

plt.show()


# Histogram of Heart Rate

plt.figure(figsize=(8, 5))

plt.hist(
    df["Heart_Rate"],
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of Heart Rate")

plt.xlabel("Heart Rate")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("heart_rate_distribution.png")

plt.show()


# Histogram of BMI

plt.figure(figsize=(8, 5))

plt.hist(
    df["BMI"],
    bins=8,
    edgecolor="black"
)

plt.title("Distribution of BMI")

plt.xlabel("BMI")

plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig("bmi_distribution.png")

plt.show()


# =====================================================
# 10. ANALYTICAL OBSERVATIONS
# =====================================================

print("\n========== ANALYTICAL OBSERVATIONS ==========")

highest_sugar = df["Sugar_Level"].max()

highest_bmi = df["BMI"].max()

highest_cholesterol = df["Cholesterol"].max()

print(
    "1. Highest recorded sugar level:",
    highest_sugar
)

print(
    "2. Highest recorded BMI:",
    highest_bmi
)

print(
    "3. Highest recorded cholesterol:",
    highest_cholesterol
)

print(
    "4. Hospital-wise patient counts were calculated."
)

print(
    "5. IQR and Z-score methods were used for outlier detection."
)

print(
    "6. Mean and median were compared for all medical parameters."
)

print(
    "7. Histograms were generated for important health indicators."
)


print("\n========== PROGRAM COMPLETED ==========")