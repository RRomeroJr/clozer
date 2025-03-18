import pandas as pd

# Read the parquet file
df = pd.read_parquet('example_data.parquet')

# Now you can work with the data as a pandas DataFrame
print(df.head())