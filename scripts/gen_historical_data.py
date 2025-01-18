#!/usr/bin/env python3

"""
gen_historical_data_ibkr.py

Connects to IBKR (TWS or Gateway) via ib_insync and fetches daily historical data for IONQ,
saving to CSV in data/historical_price_data.csv

Prerequisites:
- IBKR TWS or Gateway running with correct ports.
- 'ib_insync' package installed: pip install ib_insync

Adjust:
- HOST, PORT, CLIENT_ID as needed.
- TICKER, DURATION, BAR_SIZE, etc. to your preferences.
"""

import os
import pandas as pd
from ib_insync import IB, Stock

# Connection parameters
HOST = '127.0.0.1'
PORT = 7496  # TWS port (or 4001 for Gateway)
CLIENT_ID = 1

TICKER = 'IONQ'
OUTPUT_CSV = './data/historical_price_data.csv'

# Historical data parameters
DURATION_STR = '2 Y'       # e.g. last 2 years
BAR_SIZE = '1 day'         # daily bars
WHAT_TO_SHOW = 'TRADES'    # can be 'MIDPOINT', 'BID', 'ASK', etc.
USE_RTH = True             # regular trading hours or not

def main():
    ib = IB()
    ib.connect(HOST, PORT, CLIENT_ID)

    contract = Stock(TICKER, 'SMART', 'USD')
    # Qualify contract to ensure conId is resolved
    ib.qualifyContracts(contract)

    print(f"Requesting historical data for {TICKER}...")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=DURATION_STR,
        barSizeSetting=BAR_SIZE,
        whatToShow=WHAT_TO_SHOW,
        useRTH=USE_RTH,
        formatDate=1,
        keepUpToDate=False
    )

    if not bars:
        print("No data returned. Check symbol or subscription.")
        ib.disconnect()
        return

    df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'barCount', 'WAP'])
    # Convert date to string or keep as datetime
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Historical data saved to {OUTPUT_CSV} with {len(df)} rows.")

    ib.disconnect()

if __name__ == '__main__':
    main()
