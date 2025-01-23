#!/usr/bin/env python3

"""
advanced_time_spread.py

Analyzes multiple front–back expiry pairs for IonQ,
fetching actual implied vol from IBKR for each strike (calls or puts),
building time-spread (calendar) trades, 
running GARCH + bubble-pop Monte Carlo to revalue day-by-day,
and producing multiple 3D surfaces (one per (front, back) pair)
plus a final table of best combos across all pairs & strikes.

Careful: If IonQ is illiquid, many expiries or strikes may have no quotes/IV.
"""

import os
import sys
import math
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ib_insync import IB, Stock, Option, ContractDetails, util
from arch import arch_model
from scipy.stats import norm

##############################################################################
# CONFIG
##############################################################################

HOST = "127.0.0.1"
PORT = 7496
CLIENT_ID = 999

DATA_FOLDER = "./data"
HIST_CSV  = os.path.join(DATA_FOLDER, "historical_price_data.csv")
FUND_CSV  = os.path.join(DATA_FOLDER, "fundamentals.csv")
MACRO_CSV = os.path.join(DATA_FOLDER, "macro_data.csv")

N_MONTE_CARLO_PATHS = 5000
SIM_DAYS = 365

STRIKE_RANGE = 0.20  # ±20% around spot
SLEEP_FETCH  = 2   # seconds for modelGreeks data
MAX_PAIRS    = 10     # how many front–back pairs do we analyze at most?

USE_CALLS_INSTEAD_OF_PUTS = False  # if True => calls time spread, else puts

##############################################################################
# 1) Data loading
##############################################################################

def load_historical_data():
    path = HIST_CSV
    if not os.path.exists(path):
        print(f"ERROR: No historical CSV at {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.set_index("date", inplace=True)
    df = df[["open","high","low","close","volume"]]
    return df

def load_fundamentals():
    if not os.path.exists(FUND_CSV):
        print("WARNING: fundamentals not found.")
        return pd.DataFrame()
    return pd.read_csv(FUND_CSV)

def load_macro():
    if not os.path.exists(MACRO_CSV):
        print("WARNING: macro not found.")
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

##############################################################################
# 2) Bubble pop + RFR
##############################################################################

def compute_shock_params(fund_df, macro_df):
    daily_shock_prob = 0.001
    shock_mean = 0.3
    shock_std  = 0.1
    if not fund_df.empty:
        row = fund_df.iloc[0]
        p2rev = row.get("TTMPR2REV",1.0)
        if p2rev>50:
            daily_shock_prob += 0.001*(p2rev-50)
            shock_mean       += 0.05*(p2rev-50)/50
        ttmniac = row.get("TTMNIAC",0)
        if ttmniac<0:
            daily_shock_prob += 0.0005*abs(ttmniac)/100
            shock_mean       += 0.01*abs(ttmniac)/1000
        proj_eps = row.get("FORECAST_ProjEPS",0)
        if proj_eps<0:
            daily_shock_prob += 0.0002*abs(proj_eps)
            shock_mean       += 0.02*abs(proj_eps)
    if not macro_df.empty:
        last_row = macro_df.iloc[-1]
        fed = last_row.get("fed_funds_rate",4.0)
        if fed>1:
            fed/=100.0
        if fed>0.03:
            daily_shock_prob += 0.0002*(fed*100-3)
            shock_mean       += 0.01*(fed*100-3)
    daily_shock_prob = min(max(daily_shock_prob,0),0.05)
    shock_mean       = min(max(shock_mean,0),0.95)
    return daily_shock_prob, shock_mean, shock_std

def derive_rfr(macro_df):
    if macro_df.empty:
        return 0.04
    last_row = macro_df.iloc[-1]
    val = last_row.get("fed_funds_rate",4.0)
    if val>1:
        val/=100.0
    return val

##############################################################################
# 3) GARCH & Monte Carlo
##############################################################################

def fit_garch_volatility(df):
    ret = df["close"].pct_change().dropna()*100
    am = arch_model(ret, p=1, q=1, mean="constant", vol="GARCH", dist="normal")
    res = am.fit(update_freq=0, disp="off")
    fc = res.forecast(horizon=SIM_DAYS, reindex=False)
    var_arr = fc.variance.values[-1]
    vol_arr = np.sqrt(var_arr)/100.0
    return vol_arr

def random_shock(px, prob, mean, std):
    if np.random.rand()<prob:
        s = np.random.normal(mean, std)
        s = max(0,min(s,0.99))
        px*= (1-s)
    return px

def mc_sim(spot, mu, vol_path, shock_p, shock_m, shock_s,
           days=SIM_DAYS, n_sims= N_MONTE_CARLO_PATHS):
    sims = np.zeros((n_sims, days))
    sims[:,0] = spot
    if len(vol_path)<days:
        vol_path = np.tile(vol_path, math.ceil(days/len(vol_path)))
    for t in range(1, days):
        v_t = vol_path[t]
        z   = np.random.normal(0,1,n_sims)
        drift = (mu -0.5*(v_t*v_t))
        sims[:,t] = sims[:,t-1]*np.exp(drift + v_t*z)
        for i in range(n_sims):
            sims[i,t] = random_shock(sims[i,t], shock_p, shock_m, shock_s)
    return sims  # shape=(n_sims, days)

##############################################################################
# 4) Black-Scholes
##############################################################################

def bs_option(S, K, T, r, sigma, is_call):
    if S<=0 or K<=0 or T<=0 or sigma<=0:
        # fallback => intrinsic
        payoff = (S-K if is_call else K-S)
        return max(payoff,0)
    from math import log, sqrt, exp
    d1 = (log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if is_call:
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

##############################################################################
# 5) Calendar spread daily reval
##############################################################################

def value_calendar(
    S, 
    K,
    t,
    front_exp_days,
    back_exp_days,
    front_iv,
    back_iv,
    r,
    net_debit,
    is_call
):
    """
    Revalue calendar on day t => PnL.
    front expires front_exp_days from last_date, back => back_exp_days.
    front_iv, back_iv => baseline implied vol.
    net_debit => cost at inception.
    """
    if t>=front_exp_days:
        front_val= 0.0
    else:
        T_f = (front_exp_days - t)/365.0
        front_val = bs_option(S, K, T_f, r, front_iv, is_call)

    if t>=back_exp_days:
        back_val= 0.0
    else:
        T_b = (back_exp_days - t)/365.0
        back_val = bs_option(S, K, T_b, r, back_iv, is_call)

    return (back_val - front_val) - net_debit

##############################################################################
# 6) 3D surface: day vs strike => average PnL
##############################################################################

def calendar_3d_surface(
    sim_px,
    strikes,
    front_days_arr, # array of front expiry (days)
    back_days_arr,  # array of back expiry (days)
    front_iv_arr,
    back_iv_arr,
    net_debit_arr,
    r,
    is_call
):
    n_sims, sim_days = sim_px.shape
    n_strikes = len(strikes)
    Z = np.zeros((sim_days, n_strikes))

    for i,K in enumerate(strikes):
        fd = front_days_arr[i]
        bd = back_days_arr[i]
        fiv= front_iv_arr[i]
        biv= back_iv_arr[i]
        ndeb= net_debit_arr[i]
        for t in range(sim_days):
            vals= np.zeros(n_sims)
            for p in range(n_sims):
                S_t = sim_px[p,t]
                vals[p] = value_calendar(S_t, K, t, fd, bd, fiv, biv, r, ndeb, is_call)
            Z[t,i] = vals.mean()
    return Z

##############################################################################
# MAIN
##############################################################################

def main():
    print("=== IonQ Multi-Expiry Time Spread Analysis ===")
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print("Connected to IBKR")

    # 1) Load data
    hist_df = load_historical_data()
    fund_df = load_fundamentals()
    macro_df= load_macro()
    merged_df= merge_data(hist_df, fund_df, macro_df)
    final_df = add_technical_features(merged_df)
    last_date= final_df.index[-1].date()
    print(f"Hist ends {last_date}, shape={final_df.shape}")

    # 2) bubble + rfr
    shock_p, shock_m, shock_s = compute_shock_params(fund_df, macro_df)
    rfr= derive_rfr(macro_df)
    print(f"shock_p={shock_p:.3f}, shock_m={shock_m:.2f}, shock_s={shock_s:.2f}, rfr={rfr*100:.2f}%")

    # 3) garch => sim
    vol_path= fit_garch_volatility(final_df)
    df2= final_df.copy()
    df2["log_ret"]= np.log(df2["close"]/df2["close"].shift(1))
    df2.dropna(inplace=True)
    mu= df2["log_ret"].mean()
    spot= final_df["close"].iloc[-1]
    print(f"spot={spot:.2f}, mu={mu:.5f}")

    sim_prices= mc_sim(spot, mu, vol_path, shock_p, shock_m, shock_s)
    print(f"MC sim shape= {sim_prices.shape}")

    # 4) chain => gather all future expiries
    stock = Stock("IONQ", "SMART", "USD")
    contract = ib.reqContractDetails(stock)[0].contract
    chain_params= ib.reqSecDefOptParams(contract.symbol,"",contract.secType, contract.conId)
    if not chain_params:
        print("No chain info. Exiting.")
        return

    chain_smart= None
    for cp in chain_params:
        if cp.exchange=="SMART":
            chain_smart= cp
            break
    if not chain_smart:
        print("No SMART chain. Exiting.")
        return

    all_exps= sorted(chain_smart.expirations)
    def parse_exp(s):
        return datetime.datetime.strptime(s, "%Y%m%d").date()
    future_exps= [e for e in all_exps if parse_exp(e)> last_date]
    if len(future_exps)<2:
        print("Not enough future expiries => can't do multi-pair analysis.")
        return

    # We form pairs
    all_pairs= []
    for i in range(len(future_exps)-1):
        for j in range(i+1, len(future_exps)):
            front= future_exps[i]
            back = future_exps[j]
            all_pairs.append((front, back))
    # maybe limit to MAX_PAIRS for demonstration
    all_pairs= all_pairs[:MAX_PAIRS]
    print(f"Pairs to analyze: {all_pairs}")

    # define call/put
    is_call= USE_CALLS_INSTEAD_OF_PUTS
    right_char= "C" if is_call else "P"

    def fetch_model_iv(opt: Option):
        ib.reqMarketDataType(1)
        tkr= ib.reqMktData(opt, "100,106", False, False)
        print(f"Ticker: {tkr}")
        ib.sleep(SLEEP_FETCH)
        iv= float('nan')
        if tkr.modelGreeks and tkr.modelGreeks.impliedVol is not None:
            iv= tkr.modelGreeks.impliedVol
        ib.cancelMktData(opt)
        return iv

    def fetch_mid_price(opt: Option) -> float:
        ib.reqMarketDataType(1)
        tkr= ib.reqMktData(opt, "100,106", False, False)
        print(f"Ticker: {tkr}")
        ib.sleep(SLEEP_FETCH)
        b,a= tkr.bid, tkr.ask
        ret= float('nan')
        if b and a and b>0 and a>0 and b!=-1 and a!=-1:
            ret= 0.5*(b+a)
        elif tkr.last and tkr.last>0:
            ret= tkr.last
        elif tkr.close and tkr.close>0:
            ret= tkr.close
        ib.cancelMktData(opt)
        return ret

    # We'll store the results for each pair => big dict for building surfaces
    pair_data = []

    # We'll also store best combos across pairs for final summary
    best_rows= []

    fig = plt.figure(figsize=(10*len(all_pairs), 6))  # wide figure for subplots
    # We'll have one 3D subplot per pair
    n_subplots = len(all_pairs)

    for pair_idx,(fexp_str,bexp_str) in enumerate(all_pairs, start=1):
        fexp_dt= parse_exp(fexp_str)
        bexp_dt= parse_exp(bexp_str)
        front_days= (fexp_dt - last_date).days
        back_days = (bexp_dt - last_date).days
        print(f"Pair {pair_idx}: front={fexp_str} => {front_days} days, back={bexp_str} => {back_days} days")

        # gather near money strikes
        min_strk= spot*(1- STRIKE_RANGE)
        max_strk= spot*(1+ STRIKE_RANGE)
        cand_strikes= sorted(s for s in chain_smart.strikes if min_strk<=s<=max_strk)

        row_list=[]
        for K in cand_strikes:
            # build front/back
            c_front= Option("IONQ", fexp_str, K, right_char, "SMART")
            c_back = Option("IONQ", bexp_str, K, right_char, "SMART")
            # qualify
            try:
                cdf= ib.reqContractDetails(c_front)
                cdb= ib.reqContractDetails(c_back)
            except Exception as e:
                print(f"Error fetching contract details: {e}")
                continue
            if not cdf or not cdb:
                continue
            # fetch iv
            iv_f= fetch_model_iv(c_front)
            iv_b= fetch_model_iv(c_back)
            if math.isnan(iv_f) or math.isnan(iv_b):
                continue
            # fetch mid
            prem_f= fetch_mid_price(c_front)
            prem_b= fetch_mid_price(c_back)
            if math.isnan(prem_f) or math.isnan(prem_b):
                continue
            net_debit= prem_b - prem_f
            if net_debit<0:
                # skip credit spreads, user wants debit only
                continue
            row_list.append({
                "strike": K,
                "front_iv": iv_f,
                "back_iv":  iv_b,
                "front_prem": prem_f,
                "back_prem":  prem_b,
                "net_debit":  net_debit
            })
        if not row_list:
            print(f"No valid strikes for pair (f={fexp_str}, b={bexp_str}). Skipping.")
            continue
        pair_df= pd.DataFrame(row_list).sort_values("strike")
        if pair_df.empty:
            print(f"Empty pair_df for {fexp_str} {bexp_str}")
            continue

        # build arrays for 3D
        strikes= pair_df["strike"].values
        fiv   = pair_df["front_iv"].values
        biv   = pair_df["back_iv"].values
        ndeb  = pair_df["net_debit"].values
        # days
        fdays_arr= np.array([front_days]*len(strikes))
        bdays_arr= np.array([back_days]*len(strikes))

        # build 3D array => day vs strike => avg PnL
        Z= calendar_3d_surface(
            sim_px= sim_prices,
            strikes= strikes,
            front_days_arr= fdays_arr,
            back_days_arr= bdays_arr,
            front_iv_arr= fiv,
            back_iv_arr= biv,
            net_debit_arr= ndeb,
            r= rfr,
            is_call= is_call
        )

        # find best combos (strike, day)
        best_local_pnl= -9999999
        best_local_day= 0
        best_local_strike= None
        sim_days= Z.shape[0]
        for i,Kv in enumerate(strikes):
            arr= Z[:,i]
            local_max= arr.max()
            loc_day  = arr.argmax()
            if local_max>best_local_pnl:
                best_local_pnl= local_max
                best_local_day= loc_day
                best_local_strike= Kv

        best_rows.append({
            "front_exp": fexp_str,
            "back_exp":  bexp_str,
            "strike":    best_local_strike,
            "best_pnl":  best_local_pnl,
            "best_day":  best_local_day
        })

        # plot on subplot
        ax= fig.add_subplot(1, n_subplots, pair_idx, projection='3d')
        day_axis= np.arange(sim_days)
        X, Y= np.meshgrid(day_axis, strikes, indexing='ij')
        surf= ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        ax.set_title(f"({fexp_str} -> {bexp_str})")
        ax.set_xlabel("Day Index")
        ax.set_ylabel("Strike")
        ax.set_zlabel("Avg PnL")

        pair_data.append((fexp_str, bexp_str, pair_df, Z))

    plt.tight_layout()
    plt.show()

    # final summary table
    best_df= pd.DataFrame(best_rows).sort_values("best_pnl", ascending=False)
    print("=== Best combos across all pairs & strikes ===")
    print(best_df.head(10).to_string(index=False))

    # Done
    ib.disconnect()
    print("Disconnected. Done.")

if __name__=="__main__":
    main()