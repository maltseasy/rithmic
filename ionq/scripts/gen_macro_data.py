#!/usr/bin/env python3

"""
gen_macro_data_fred.py

Fetches real macro data (e.g. the Federal Funds Effective Rate: DFF) from FRED,
and saves it as data/macro_data.csv.

Requires:
- 'fredapi' package: pip install fredapi
- A valid FRED API key. See: https://research.stlouisfed.org/docs/api/api_key.html

You can replace 'DFF' with any other FRED series ID that is relevant for your model.
"""

import os
import pandas as pd
from fredapi import Fred

FRED_API_KEY = "cec79dd72150c9f617aec63ce378644a"  # <-- Place your real key here
SERIES_ID = "DFF"  # Federal Funds Effective Rate
OUTPUT_CSV = "./data/macro_data.csv"

def main():
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series(SERIES_ID)
    # data is a pandas Series with datetime index, values are the fed funds rate

    df = pd.DataFrame(data, columns=['fed_funds_rate'])
    df.index.name = 'date'
    df.reset_index(inplace=True)
    
    # We have date and fed_funds_rate columns
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Macro data from FRED (series: {SERIES_ID}) saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
