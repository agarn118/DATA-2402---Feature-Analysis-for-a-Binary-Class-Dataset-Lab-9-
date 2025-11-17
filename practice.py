import pandas as pd


def split_by_class(df):
    # Use boolean indexing to filter rows
    df_class_0 = df[df['TumorType'] == 0]
    df_class_1 = df[df['TumorType'] == 1]
    
    return df_class_0, df_class_1




data = r'C:\Users\ashto\OneDrive\Desktop\Lab 9-10\DATA-2402---Feature-Analysis-for-a-Binary-Class-Dataset-Lab-9-\cancer_historical (1).csv'
df = pd.read_csv(data)

df_class_0, df_class_1 = split_by_class(df)
print(df_class_0)
print(df_class_1)