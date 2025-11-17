import pandas as pd

# Task 1 - Function Writing
def split_by_class(df):
    """
    Receives a DataFrame where the last column is a binary class (0/1).
    Returns two DataFrames: Class_0 and Class_1.
    """
    # Use boolean indexing to filter rows
    df_class_0 = df[df['TumorType'] == 0].reset_index(drop=True)
    df_class_1 = df[df['TumorType'] == 1].reset_index(drop=True)
    
    return df_class_0, df_class_1


# Task 2 - Data Validation
def validate_data(df):
    """
    Checks basic assumptions about the dataset.
    Raises ValueError with a clear message if something is wrong.
    """
    label_col = df.columns[-1]
    feature_cols = df.columns[:-1]
    # The last column has exactly two unique values 0 or 1.
    unique_labels = set(df[label_col].dropna().unique())
    if unique_labels != {0, 1}:
        raise ValueError("The last column has values other than 0 or 1.")
    
    # None of the 9 feature columns are entirely missing (all NaN)
    all_nan = df[feature_cols].isna().all()
    if all_nan.any():
        bad_cols = list(all_nan[all_nan].index)    
        raise ValueError(f"The following feature columns are entirely missing (all NaN): {bad_cols}.")
    
    # There  are no duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        raise ValueError("There are duplicate rows of data.")
    
    return True     # Passed all checks


# Task 3 - Statistical Comparison
def compute_stats(df):
    """
    Computes the mean
    """


# -----------------------------MAIN SCRIPT-----------------------------
data = r'C:\Users\ashto\OneDrive\Desktop\Lab 9-10\DATA-2402---Feature-Analysis-for-a-Binary-Class-Dataset-Lab-9-\cancer_historical (1).csv'
df = pd.read_csv(data)

# Validate the data
try:
    validate_data(df)
except ValueError as e:
    print(f"Validation Error: {e}")


# Split by Tumor Type
df_class_0, df_class_1 = split_by_class(df)

# Statistical Comparison
stats_class_0 = compute_stats(df_class_0)
# stats_class_1 = compute_stats(df_class_1)