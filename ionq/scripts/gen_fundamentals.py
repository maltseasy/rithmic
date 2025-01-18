#!/usr/bin/env python3

"""
generate_fundamentals_ibkr.py

Requests IONQ fundamental data from IBKR using 'reqFundamentalData' call,
then parses the returned XML using an XML parser instead of regex.

Extracts key fields from the <Ratios> and <ForecastData> sections, 
saves them in structured form to data/fundamentals.csv

Prerequisites:
- IBKR TWS or Gateway running with correct market data / fundamentals subscription.
- 'ib_insync' installed: pip install ib_insync
"""

import os
import pandas as pd
import xml.etree.ElementTree as ET  # The built-in XML parser
from ib_insync import IB, Stock

HOST = '127.0.0.1'
PORT = 7496   # or 4001 if you're using IB Gateway
CLIENT_ID = 2

TICKER = 'IONQ'
OUTPUT_CSV = './data/fundamentals.csv'

def parse_fundamentals_xml(xml_str):
    """
    Use xml.etree.ElementTree to parse the returned XML.
    Extract ratio fields from <Ratios> and <ForecastData>.
    Return a dictionary with parsed values.
    """
    root = ET.fromstring(xml_str)
    data = {}

    # 1) Parse <Ratios> section
    #    Each <Ratio> has attribute 'FieldName' and text content, e.g.:
    #    <Ratio FieldName="MKTCAP" Type="N">6028.68800</Ratio>
    #    We'll store them in data dict with keys matching the FieldName, e.g. data["MKTCAP"] = 6028.688
    for ratio_el in root.findall(".//Ratios//Ratio"):
        field_name = ratio_el.attrib.get("FieldName", "")
        text_value = (ratio_el.text or "").strip()
        # Convert to float if possible
        if text_value and text_value not in ["-99999.99000", "-99999.99"]:  # IBKR uses sentinel for N/A
            try:
                data[field_name] = float(text_value)
            except ValueError:
                data[field_name] = text_value  # fallback if not numeric

    # 2) Parse <ForecastData> section
    #    The forecast data is nested like:
    #    <ForecastData ...>
    #       <Ratio FieldName="TargetPrice" Type="N">
    #         <Value PeriodType="CURR">34.05000</Value>
    #       </Ratio>
    #    We'll gather the <Value PeriodType="CURR">...<Value> text.
    forecast_el = root.find(".//ForecastData")
    if forecast_el is not None:
        for ratio_el in forecast_el.findall(".//Ratio"):
            field_name = ratio_el.attrib.get("FieldName", "")
            # Look for a child <Value PeriodType="CURR">
            val_el = ratio_el.find(".//Value[@PeriodType='CURR']")
            if val_el is not None and val_el.text:
                val_text = val_el.text.strip()
                if val_text not in ["-99999.99000", "-99999.99"]:
                    try:
                        data[f"FORECAST_{field_name}"] = float(val_text)
                    except ValueError:
                        data[f"FORECAST_{field_name}"] = val_text

    # 3) Optionally parse other top-level fields if you like, e.g. <CoIDs>, <CoGeneralInfo>, etc.
    #    For example, to get the CompanyName from <CoID Type="CompanyName">IONQ Inc</CoID>:
    coid_company = root.find(".//CoID[@Type='CompanyName']")
    if coid_company is not None and coid_company.text:
        data["CompanyName"] = coid_company.text.strip()

    #    To parse the "SharesOut" if needed:
    shares_out_el = root.find(".//CoGeneralInfo//SharesOut")
    if shares_out_el is not None and shares_out_el.text:
        try:
            data["SharesOutstanding"] = float(shares_out_el.text.strip())  # total shares
        except ValueError:
            pass

        # Could also parse "TotalFloat" from the attribute
        total_float = shares_out_el.attrib.get("TotalFloat")
        if total_float:
            try:
                data["FloatShares"] = float(total_float)
            except ValueError:
                pass

    #    Similarly, employees from <Employees>:
    employees_el = root.find(".//CoGeneralInfo//Employees")
    if employees_el is not None and employees_el.text:
        try:
            data["Employees"] = float(employees_el.text.strip())
        except ValueError:
            pass

    return data


def main():
    ib = IB()
    ib.connect(HOST, PORT, CLIENT_ID)

    contract = Stock(TICKER, 'SMART', 'USD')
    ib.qualifyContracts(contract)

    # Request fundamental data from IBKR
    # Report types: 'ReportSnapshot', 'ReportsFinSummary', 'ReportRatios', etc.
    fundamentals_xml = ib.reqFundamentalData(contract, reportType='ReportSnapshot')
    if not fundamentals_xml:
        print("No fundamental data returned. Check subscription or symbol.")
        ib.disconnect()
        return

    # Parse
    parsed_data = parse_fundamentals_xml(fundamentals_xml)
    if not parsed_data:
        print("Could not parse fundamental fields from the returned XML. Check patterns or data structure.")
        # If you want to debug, you can print the raw XML here
        # print(fundamentals_xml)
    else:
        print("Parsed fundamentals:\n", parsed_data)

    # Convert to DataFrame (single row)
    df = pd.DataFrame([parsed_data])

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Fundamentals saved to {OUTPUT_CSV}")
    ib.disconnect()

if __name__ == '__main__':
    main()
