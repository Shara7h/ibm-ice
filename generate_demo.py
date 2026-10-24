import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    " First Name ": ["John", "Jane", None, "Bob", "Alice"] * 200,
    "Date_of_Purchase": ["2023-01-01", "01/02/2023", "not a date", None, "2023/05/06"] * 200,
    " Sales ": np.random.normal(100, 20, 1000).tolist(),
    "Age": np.random.randint(20, 60, 1000).tolist()
})
df.loc[50, " Sales "] = 99999
df.loc[100, "Age"] = -5
df = pd.concat([df, df.iloc[:50]])  # duplicates
df.to_csv("sample_data/messy_sales_data.csv", index=False)
print("Demo dataset created:", len(df), "rows")