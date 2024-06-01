import pandas as pd
from spam import spam_data_file_path

df = pd.read_csv(spam_data_file_path, sep='\t', header=None, names=['Label', 'Text'])
print(df.head())

print(df.tail())

print(df["Label"].value_counts())
