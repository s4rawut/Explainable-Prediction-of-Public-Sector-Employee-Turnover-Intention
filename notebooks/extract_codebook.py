import pandas as pd
import sys

file_path = '../data/FEVS2022_PRDF_CSV/2022_OPM_FEVS_PRDF_Codebook_r2.xlsx'
try:
    xl = pd.ExcelFile(file_path)
    print("Sheet names:", xl.sheet_names)
    
    df = xl.parse(xl.sheet_names[0])
    print("\nColumns in first sheet:", df.columns.tolist())
    
    print("\nSample rows:")
    print(df.head())
    
    df.to_csv('../data/codebook_extracted.csv', index=False)
    print("\nSaved extracted codebook to data/codebook_extracted.csv")
except Exception as e:
    print("Error:", e)
