#!/usr/bin/env python3

"""
debug_qualify_contract.py

A small script to check if IBKR recognizes a specific IonQ option contract:
  - Expiry in 'YYYY-MM-DD' format (not 'YYYYMMDD')
  - A chosen strike price
  - Put or Call

Run:
  python debug_qualify_contract.py

Then see if reqContractDetails returns a valid contract or empty.
If empty, IBKR doesn't actually see that contract as valid for your account/data subscription.
"""

from ib_insync import IB, Option

HOST = '127.0.0.1'
PORT = 7496
CLIENT_ID = 1234

def main():
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    expiry_str = '20250214'   # or any monthly date in YYYYMMDD
    strike = 37.5
    put_contract = Option(symbol='IONQ',
                          lastTradeDateOrContractMonth=expiry_str,
                          strike=strike,
                          right='P',
                          exchange='SMART')

    print(f"Testing IonQ option: {put_contract}")
    details = ib.reqContractDetails(put_contract)
    if not details:
        print(f"No contract details returned => IBKR does NOT recognize {put_contract}.")
    else:
        print(f"Contract details for {put_contract}:")
        for d in details:
            print(" ", d)

        # Optionally request market data to see if we can get a quote
        ticker = ib.reqMktData(put_contract, "", False, False)
        ib.sleep(1)  # wait for data
        print(f"Bid={ticker.bid}, Ask={ticker.ask}, Last={ticker.last}")

    ib.disconnect()

if __name__=='__main__':
    main()
