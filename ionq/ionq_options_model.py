#!/usr/bin/env python3

"""
ionq_options_model.py
Usage:
  python ionq_options_model.py
"""

import os
import sys
import math
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from arch import arch_model

from ib_insync import IB, Stock, Option

###############################################################################
# CONFIG
###############################################################################

HOST = "127.0.0.1"
PORT = 7496
CLIENT_ID = 999

DATA_FOLDER = "./data"
HIST_CSV = os.path.join(DATA_FOLDER, "historical_price_data.csv")
FUND_CSV = os.path.join(DATA_FOLDER, "fundamentals.csv")
MACRO_CSV = os.path.join(DATA_FOLDER, "macro_data.csv")

N_MONTE_CARLO_PATHS = 5000
SIM_DAYS = 365
SLEEP_BETWEEN_REQUESTS = 0.3

EXPIRIES_TO_CHECK = 8  # how many future monthly expiries to examine
STRIKE_RANGE_PCT = 0.20  # ±20% around spot

###############################################################################
# 1) Data Loading
###############################################################################

def load_historical_data():
    path = HIST_CSV
    if not os.path.exists(path):
        print(f"ERROR: No historical data at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]

def load_fundamentals():
    if not os.path.exists(FUND_CSV):
        print("WARNING: No fundamentals found.")
        return pd.DataFrame()
    return pd.read_csv(FUND_CSV)

def load_macro():
    if not os.path.exists(MACRO_CSV):
        print("WARNING: No macro found.")
        return pd.DataFrame()
    df = pd.read_csv(MACRO_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df

def merge_data(hist_df, fund_df, macro_df):
    df = hist_df.copy()
    if not macro_df.empty:
        macro_df = macro_df.reindex(df.index).ffill()
        for col in macro_df.columns:
            df[col] = macro_df[col]
    if not fund_df.empty:
        # Just embed the single row or first row
        for c in fund_df.columns:
            df[c] = fund_df[c].iloc[0]
    df.dropna(inplace=True)
    return df

def add_technical_features(df):
    df = df.copy()
    df["pct_change"] = df["close"].pct_change()
    df["MA_20"] = df["close"].rolling(20).mean()
    df["Vol_20"] = df["pct_change"].rolling(20).std()
    df.dropna(inplace=True)
    return df

###############################################################################
# 2) Bubble pop & RFR
###############################################################################

def compute_shock_params(fund_df, macro_df):
    daily_shock_prob = 0.001
    shock_mean = 0.3
    shock_std = 0.1
    if not fund_df.empty:
        row = fund_df.iloc[0]
        p2rev = row.get("TTMPR2REV",1.0)
        if p2rev>50:
            daily_shock_prob += 0.001*(p2rev-50)
            shock_mean += 0.05*(p2rev-50)/50
        ttmniac = row.get("TTMNIAC",0)
        if ttmniac<0:
            daily_shock_prob += 0.0005*abs(ttmniac)/100
            shock_mean += 0.01*abs(ttmniac)/1000
        proj_eps = row.get("FORECAST_ProjEPS",0)
        if proj_eps<0:
            daily_shock_prob += 0.0002*abs(proj_eps)
            shock_mean += 0.02*abs(proj_eps)
    if not macro_df.empty:
        last_row = macro_df.iloc[-1]
        fed = last_row.get("fed_funds_rate",4.0)
        if fed>1:
            fed/=100.0
        if fed>0.03:
            daily_shock_prob += 0.0002*(fed*100-3)
            shock_mean += 0.01*(fed*100-3)
    daily_shock_prob = min(max(daily_shock_prob,0),0.05)
    shock_mean = min(max(shock_mean,0),0.95)
    return daily_shock_prob, shock_mean, shock_std

def derive_rfr(macro_df):
    if macro_df.empty:
        return 0.04
    last_row = macro_df.iloc[-1]
    val = last_row.get("fed_funds_rate",4.0)
    if val>1:
        val/=100.0
    return val

###############################################################################
# 3) GARCH & Monte Carlo
###############################################################################

def fit_garch_volatility(df):
    ret = df["close"].pct_change().dropna()*100
    am = arch_model(ret, p=1,q=1, mean="constant", vol="GARCH", dist="normal")
    res = am.fit(update_freq=0, disp="off")
    fc = res.forecast(horizon=SIM_DAYS, reindex=False)
    var_arr = fc.variance.values[-1]
    vol_arr = np.sqrt(var_arr)/100.0
    return vol_arr

def random_shock(px, prob, mean, std):
    if np.random.rand()<prob:
        s = np.random.normal(mean, std)
        s = max(0, min(s,0.99))
        px*= (1-s)
    return px

def mc_sim(spot, mu, vol_path, shock_p, shock_m, shock_s,
           days=SIM_DAYS, n_sims=N_MONTE_CARLO_PATHS):
    sims = np.zeros((n_sims, days))
    sims[:,0] = spot
    if len(vol_path)<days:
        vol_path = np.tile(vol_path, math.ceil(days/len(vol_path)))
    for t in range(1, days):
        vt = vol_path[t]
        z = np.random.normal(0,1,n_sims)
        drift = (mu -0.5*(vt**2))
        sims[:,t] = sims[:,t-1]*np.exp(drift + vt*z)
        for i in range(n_sims):
            sims[i,t] = random_shock(sims[i,t], shock_p, shock_m, shock_s)
    return sims

###############################################################################
# 4) BS Put & Evaluate
###############################################################################

def bs_put(S, K, T, r, sigma):
    if S<=0 or K<=0 or T<=0 or sigma<=0:
        return 0.0
    from math import log, sqrt, exp
    d1 = (log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def evaluate_long_put(sim_px, strike, expiry_idx, premium):
    if expiry_idx>=sim_px.shape[1]:
        expiry_idx=sim_px.shape[1]-1
    finals = sim_px[:, expiry_idx]
    payoff = np.maximum(strike-finals,0)
    return payoff - premium

###############################################################################
# 5) 3D Surface Plot (blocking)
###############################################################################

def plot_3d_pnl(sim_px, strike, premium, resolution=120):
    days = sim_px.shape[1]
    p_min, p_max = sim_px.min(), sim_px.max()
    p_min = max(p_min, 0)
    px_range = np.linspace(p_min, p_max, resolution)
    Z = np.zeros((days,resolution))
    for d in range(days):
        for j, px in enumerate(px_range):
            payoff = max(strike - px,0)
            Z[d,j] = payoff - premium
    X, Y = np.meshgrid(np.arange(days), px_range, indexing="ij")
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.8)
    ax.set_title(f"IonQ Put PnL Surface: Strike={strike}, Premium={premium:.2f}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Underlying Price")
    ax.set_zlabel("PnL")
    plt.show(block=True)  # block so user sees the figure

###############################################################################
# 6) MAIN
###############################################################################

def main():
    print("=== IonQ Options Model (Strict) ===")
    ib = IB()

    error_flags = {}
    def on_error(reqId, errorCode, errorString, contract):
        if errorCode==200:
            error_flags[reqId] = True
            print(f"[Error 200] Skipping: {contract} => {errorString}")
    ib.errorEvent += on_error

    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID)
        print(f"Connected: {HOST}:{PORT}, clientId={CLIENT_ID}")

        # 1) load data
        hist_df = load_historical_data()
        fund_df = load_fundamentals()
        macro_df = load_macro()
        merged_df = merge_data(hist_df, fund_df, macro_df)
        final_df = add_technical_features(merged_df)
        print(f"Merged shape={final_df.shape}")

        # 2) bubble & rfr
        shock_p, shock_m, shock_s = compute_shock_params(fund_df, macro_df)
        rfr = derive_rfr(macro_df)
        print(f"shock_p={shock_p:.4f}, shock_m={shock_m:.3f}, shock_s={shock_s:.3f}, rfr={rfr*100:.2f}%")

        # 3) GARCH => MC
        vol_path = fit_garch_volatility(final_df)
        print(f"GARCH vol path={len(vol_path)}")
        final_df["log_ret"] = np.log(final_df["close"]/final_df["close"].shift(1))
        final_df.dropna(inplace=True)
        mu = final_df["log_ret"].mean()
        spot = final_df["close"].iloc[-1]
        print(f"Spot={spot:.2f}, mu={mu:.5f}")
        sim_px = mc_sim(spot, mu, vol_path, shock_p, shock_m, shock_s,
                        days=SIM_DAYS, n_sims=N_MONTE_CARLO_PATHS)
        print(f"MC sim shape={sim_px.shape}")

        # 4) IonQ stock => chain
        stock_contract = Stock("IONQ","SMART","USD")
        cdetails = ib.reqContractDetails(stock_contract)
        if not cdetails:
            print("No IonQ contract details => can't proceed.")
            return
        chain_params = ib.reqSecDefOptParams(cdetails[0].contract.symbol, "",
                                             cdetails[0].contract.secType,
                                             cdetails[0].contract.conId)
        if not chain_params:
            print("No chain info => can't proceed.")
            return
        chain_info = None
        for cp in chain_params:
            if cp.exchange=="SMART":
                chain_info = cp
                break
        if not chain_info:
            print("No SMART chain => can't proceed.")
            return

        all_exps = sorted(list(chain_info.expirations))
        print("All Expiries from IBKR:", all_exps)

        # 5) filter future
        last_date = final_df.index[-1].date()
        future_exps = [e for e in all_exps if datetime.datetime.strptime(e,"%Y%m%d").date()>last_date]
        future_exps.sort()
        future_exps = future_exps[:EXPIRIES_TO_CHECK]
        print("Chosen Expiries:", future_exps)

        # near-money range
        min_strike = spot*(1.0 - STRIKE_RANGE_PCT)
        max_strike = spot*(1.0 + STRIKE_RANGE_PCT)

        results = []
        for expiry_str in future_exps:
            expiry_dt = datetime.datetime.strptime(expiry_str,"%Y%m%d").date()
            days_to_expiry = (expiry_dt - last_date).days
            valid_strikes = [s for s in chain_info.strikes if s>=min_strike and s<=max_strike]
            valid_strikes.sort()

            print(f"Processing expiry={expiry_str}, near-money strikes={len(valid_strikes)} ...")
            best_strike = None
            best_pnl = -1e9
            best_prem = None

            for s in valid_strikes:
                error_flags.clear()
                opt_contract = Option("IONQ", expiry_str, s, "P", "SMART")
                # 1) check details
                cdet = ib.reqContractDetails(opt_contract)
                if not cdet:
                    # means IBKR truly doesn't see that strike or we got immediate error
                    continue

                # 2) request MktData
                ticker = ib.reqMktData(opt_contract, "", False, False)
                ib.sleep(SLEEP_BETWEEN_REQUESTS)
                if any(error_flags.values()):
                    # error 200 => skip
                    continue

                if ticker.bid is None or ticker.ask is None:
                    continue
                premium = 0.5*(ticker.bid + ticker.ask)

                # Evaluate PnL
                payoff_arr = evaluate_long_put(sim_px, s, days_to_expiry, premium)
                avg_pnl = payoff_arr.mean()
                if avg_pnl>best_pnl:
                    best_pnl=avg_pnl
                    best_strike=s
                    best_prem = premium

            print(f"For expiry={expiry_str}, best_strike={best_strike}, best avg PnL={best_pnl:.2f}")
            results.append({
                "expiry": expiry_str,
                "best_strike": best_strike,
                "expected_pnl": best_pnl
            })
            if best_strike is not None and best_prem is not None:
                # Show 3D surface blocking
                plot_3d_pnl(sim_px, best_strike, best_prem, resolution=120)

        print("\n=== Final Results ===")
        for r in results:
            print(r)

    except KeyboardInterrupt:
        print("\nUser aborted with Ctrl+C.")
    finally:
        ib.disconnect()
        print("Disconnected. Done.")

if __name__=="__main__":
    main()
