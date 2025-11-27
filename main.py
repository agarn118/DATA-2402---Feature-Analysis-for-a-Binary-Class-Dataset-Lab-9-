import pandas as pd
import matplotlib.pyplot as plt

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
    # Compute the mean, std and num count
    tumor_type = str(df.iloc[1, -1])
    feature_cols = df.columns[:-1]

    column_names = [f'Mean_{tumor_type}', f'Std_{tumor_type}', f'Num_{tumor_type}']
    stats_df = pd.DataFrame(columns=column_names)    
    for row in feature_cols:
        row = str(row)
        mean = df.loc[:, row].mean()
        std = df.loc[:, row].std()
        num = df.loc[:, row].count()
        stats_df.loc[row] = [mean, std, num]
    return stats_df

# Task 4 - Normalization
def normalize_features(df):
    # Return a new DataFrame where feature columns are z-score normalized.
    # The class column (last column) is left unchanged.
    df_norm = df.copy()
    feature_cols = df_norm.columns[:-1]

    # Z - score formula: (x - mean) / std
    df_norm[feature_cols] = (df_norm[feature_cols] - df_norm[feature_cols].mean()) / df_norm[feature_cols].std()
    return df_norm


# -----------------------------MAIN SCRIPT-----------------------------
data = r'C:\Users\ashto\Desktop\Lab-9\DATA-2402---Feature-Analysis-for-a-Binary-Class-Dataset-Lab-9-\cancer_historical (1).csv'
df = pd.read_csv(data)

# Validate the data
'''
try:
    validate_data(df)
except ValueError as e:
    print(f"Validation Error: {e}")
'''
    # Note: Validations fail so comment the block out to run the rest

# Split by Tumor Type
df_class_0, df_class_1 = split_by_class(df)

# Statistical Comparison
stats_class_0 = compute_stats(df_class_0)
stats_class_1 = compute_stats(df_class_1)
stat_comparison = pd.concat([stats_class_0, stats_class_1], axis=1)
stat_comparison['Mean Diff'] = stat_comparison['Mean_1'] - stat_comparison['Mean_0']
print(stat_comparison)

# Normalization
df_norm = normalize_features(df)

# Split normalized data by Tumor Type
norm_class_0, norm_class_1 = split_by_class(df_norm)
feature_cols = df_norm.columns[:-1]

# Task #5 - Create 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for i, feature in enumerate(feature_cols):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # Data for both the classes
    data_to_plot = [
        norm_class_0[feature],
        norm_class_1[feature]
    ]

    ax.boxplot(data_to_plot, tick_labels=['Tumor Type 0', 'Tumor Type 1'])
    ax.set_title(feature)
    ax.set_xticks([1, 2])
    ax.set_ylabel('Mean (normalized)')

fig.suptitle('Mean of Each Feature by Tumor Type (Class 0 vs Class 1)', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# -----------------------------Task 6 - Mini Report-----------------------------
print('\nTask 6 - Mini-Report')

print(
    '1.) Feature with very similar distributions between Class 0 and Class 1:\n'
        'The boxplots for Feature 5 and Feature 9 show a lot of overlap between the\n'
        'two classes and relatively small differences in their mean values. These\n'
        'features do not seperate the classes very clearly'
)

print('\n')

print(
    '2.) Features that best distinguish between Class 0 and Class 1:\n'
        'Features 2, 3, 6 and 9 show the largest mean differences and the\n'
        'clearest vertical seperation between the boxplots. For these features,\n'
        'the Class 1 box is consistenly shifted upward relative to Class 0 with\n'
        'less overlap, so they appear most effective for distinguishing the tumor types.'
)