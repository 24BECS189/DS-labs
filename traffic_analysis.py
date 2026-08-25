import pandas as pd
import glob
import os

print("\n========== SMART CITY TRAFFIC MANAGEMENT ==========")


# ---------------------------------------------------
# 1. LOAD MULTIPLE CSV FILES
# ---------------------------------------------------

files = glob.glob("junction*.csv")

print("\nCSV Files Found:")
for file in files:
    print("-", file)


dataframes = []

for file in files:

    df = pd.read_csv(file)

    # Add source file name
    df["Source_File"] = os.path.basename(file)

    dataframes.append(df)


# ---------------------------------------------------
# 2. MERGE ALL DATASETS
# ---------------------------------------------------

traffic_data = pd.concat(dataframes, ignore_index=True)

print("\n========== MERGED TRAFFIC DATA ==========")

print("\nTotal Records:", len(traffic_data))

print("\nFirst 5 Records:")
print(traffic_data.head())


# ---------------------------------------------------
# 3. CHECK MISSING VALUES
# ---------------------------------------------------

print("\n========== MISSING VALUES ==========")

missing_values = traffic_data.isnull().sum()

print(missing_values)


# ---------------------------------------------------
# 4. DETECT CORRUPTED RECORDS
# ---------------------------------------------------

print("\n========== CORRUPTED RECORDS ==========")

# Convert important columns to numeric
traffic_data["Vehicle_Count"] = pd.to_numeric(
    traffic_data["Vehicle_Count"],
    errors="coerce"
)

traffic_data["Average_Speed"] = pd.to_numeric(
    traffic_data["Average_Speed"],
    errors="coerce"
)

traffic_data["Signal_Timing"] = pd.to_numeric(
    traffic_data["Signal_Timing"],
    errors="coerce"
)

traffic_data["Accidents"] = pd.to_numeric(
    traffic_data["Accidents"],
    errors="coerce"
)

# Check invalid values
corrupted = traffic_data[
    (traffic_data["Vehicle_Count"] < 0) |
    (traffic_data["Average_Speed"] < 0) |
    (traffic_data["Signal_Timing"] < 0) |
    (traffic_data["Accidents"] < 0)
]

if corrupted.empty:
    print("No corrupted records found.")
else:
    print(corrupted)


# ---------------------------------------------------
# 5. DUPLICATE TIMESTAMPS
# ---------------------------------------------------

print("\n========== DUPLICATE TIMESTAMPS ==========")

duplicates = traffic_data[
    traffic_data.duplicated(
        subset=["Timestamp", "Location"],
        keep=False
    )
]

if duplicates.empty:
    print("No duplicate timestamps found.")
else:
    print(duplicates[
        ["Timestamp", "Location"]
    ])


# ---------------------------------------------------
# 6. COMPARE MISSING VALUE IMPUTATION
# ---------------------------------------------------

print("\n========== IMPUTATION TECHNIQUES ==========")

# Original missing values
print("\nOriginal Missing Values:")
print(
    traffic_data[
        ["Vehicle_Count", "Average_Speed"]
    ].isnull().sum()
)


# Mean Imputation
mean_data = traffic_data.copy()

mean_data["Vehicle_Count"] = mean_data[
    "Vehicle_Count"
].fillna(
    mean_data["Vehicle_Count"].mean()
)

mean_data["Average_Speed"] = mean_data[
    "Average_Speed"
].fillna(
    mean_data["Average_Speed"].mean()
)


print("\nAfter Mean Imputation:")
print(
    mean_data[
        ["Vehicle_Count", "Average_Speed"]
    ].isnull().sum()
)


# Median Imputation
median_data = traffic_data.copy()

median_data["Vehicle_Count"] = median_data[
    "Vehicle_Count"
].fillna(
    median_data["Vehicle_Count"].median()
)

median_data["Average_Speed"] = median_data[
    "Average_Speed"
].fillna(
    median_data["Average_Speed"].median()
)


print("\nAfter Median Imputation:")
print(
    median_data[
        ["Vehicle_Count", "Average_Speed"]
    ].isnull().sum()
)


# ---------------------------------------------------
# 7. USE MEDIAN IMPUTATION FOR FINAL DATA
# ---------------------------------------------------

traffic_data["Vehicle_Count"] = traffic_data[
    "Vehicle_Count"
].fillna(
    traffic_data["Vehicle_Count"].median()
)

traffic_data["Average_Speed"] = traffic_data[
    "Average_Speed"
].fillna(
    traffic_data["Average_Speed"].median()
)


# ---------------------------------------------------
# 8. REMOVE DUPLICATE RECORDS
# ---------------------------------------------------

traffic_data = traffic_data.drop_duplicates(
    subset=["Timestamp", "Location"],
    keep="first"
)


# ---------------------------------------------------
# 9. TRAFFIC DENSITY REPORT
# ---------------------------------------------------

print("\n========== TRAFFIC DENSITY REPORT ==========")

density_report = traffic_data.groupby(
    "Location"
).agg(
    Total_Vehicles=("Vehicle_Count", "sum"),
    Average_Speed=("Average_Speed", "mean"),
    Average_Signal_Timing=("Signal_Timing", "mean"),
    Total_Accidents=("Accidents", "sum")
)

print(
    density_report.round(2)
)


# ---------------------------------------------------
# 10. CONGESTION REPORT
# ---------------------------------------------------

print("\n========== CONGESTION REPORT ==========")

congestion_report = pd.crosstab(
    traffic_data["Location"],
    traffic_data["Congestion_Level"]
)

print(congestion_report)


# ---------------------------------------------------
# 11. SAVE PROCESSED DATASET
# ---------------------------------------------------

output_file = "processed_traffic_data.csv"

traffic_data.to_csv(
    output_file,
    index=False
)

print("\n========== DATA SAVING ==========")

print(
    "Processed dataset saved successfully!"
)

print("File Name:", output_file)

print("\n========== PROGRAM COMPLETED ==========")