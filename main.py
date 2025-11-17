import pandas as pd

# Task 1 - Function Writing
def split_by_class(df):
    """
    Receives a DataFrame where the last column is a binary class (0/1).
    Returns two DataFrames: Class_0 and Class_1.
    """
    # Use boolean indexing to filter rows
    df_class_0 = df[df['TumorType'] == 0]
    df_class_1 = df[df['TumorType'] == 1]
    
    return df_class_0, df_class_1

data = r'C:\Users\ashto\OneDrive\Desktop\Lab 9-10\DATA-2402---Feature-Analysis-for-a-Binary-Class-Dataset-Lab-9-\cancer_historical (1).csv'
df = pd.read_csv(data)

df_class_0, df_class_1 = split_by_class(df)
print(df_class_0)
print(df_class_1)


# Task 2 - Data Validation
def validate_data(df):
    """
    Checks basic assumptions about the dataset.
    Raises ValueError with a clear message if something is wrong.
    """
    