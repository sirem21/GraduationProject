import pandas as pd
df_raw = pd.read_csv('sample_dataset.csv')
for index, raw_row in df_raw.iterrows():
    val = str(raw_row.get('Diabetes_012', ''))
    if val in ['Diyabet', 'Prediyabet (Gizli Seker)']:
        print(f"Match found at index {index}: {val}")
    elif 'Diyabet' in val or 'Prediyabet' in val:
        print(f"Partial match at index {index}: {repr(val)}")
