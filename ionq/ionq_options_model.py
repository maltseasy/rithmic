#!/usr/bin/env python3

"""
ionq_calendar_spread_advanced.py

Implements:
 1) Automatic detection of two earliest valid expiries (front & back).
 2) Real expiry day offsets from last hist date -> front_days, back_days.
 3) Actual implied volatility from IBKR for each strike & expiry 
    (via ticker.modelGreeks.impliedVol).
 4) Choice of calls or puts (both produce a net debit calendar).
 5) Monte Carlo (GARCH + bubble pop) daily reval.
 6) Auto-select best strike by peak expected PnL. 
 7) Basic "early exit" logic: find day t that yields max PnL if we exit then.

Disclaimer: If IonQ options are illiquid, you may see -1 or no modelGreeks 
returned. TWS must show real-time quotes + model data for this to work.
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
SLEEP_FETCH = 1.0    # seconds to wait for modelGreeks after reqMktData

USE_CALLS_INSTEAD_OF_PUTS = False  # set True to do calls calendar

##############################################################################
# 1) Data loading
##############################################################################

def load_historical_data():
    if not os.path.exists(HIST_CSV):
        print(f"ERROR: No historical CSV at {HIST_CSV}")
        sys.exit(1)
    df = pd.read_csv(HIST_CSV)
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
        if fed>1:  # e.g. 4.5 => 0.045
            fed/=100.0
        if fed>0.03:
            daily_shock_prob += 0.0002*(fed*100-3)
            shock_mean += 0.01*(fed*100-3)
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
    forecast = res.forecast(horizon=SIM_DAYS, reindex=False)
    var_arr = forecast.variance.values[-1]
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
# 4) Black-Scholes for calls/puts
##############################################################################

def bs_option(S, K, T, r, sigma, is_call):
    if S<=0 or K<=0 or T<=0 or sigma<=0:
        # near expiry or invalid => do intrinsic
        payoff = (S-K if is_call else K-S)
        return max(payoff, 0)
    from math import log, sqrt, exp
    d1 = (log(S/K) + (r +0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if is_call:
        return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        # put
        return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

##############################################################################
# 5) Calendar Spread daily revaluation
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
    Return spread PnL at day t.
     - front expires at front_exp_days
     - back  expires at back_exp_days
     - front_iv, back_iv => implied vol
     - net_debit => what we paid initially (long back - short front).
    """
    # If front expired
    if t>=front_exp_days:
        front_val=0.0
    else:
        T_f = (front_exp_days - t)/365.0
        sigma_f = front_iv
        front_val = bs_option(S, K, T_f, r, sigma_f, is_call)
    
    # If back expired
    if t>=back_exp_days:
        back_val=0.0
    else:
        T_b = (back_exp_days - t)/365.0
        sigma_b = back_iv
        back_val = bs_option(S, K, T_b, r, sigma_b, is_call)
    
    # net spread value = long(back_val) - short(front_val)
    # PnL = spread_value - net_debit
    return (back_val - front_val) - net_debit

##############################################################################
# 6) Build 3D PnL: day vs strike => average
##############################################################################

def calendar_3d_surface(
    sim_px,         # shape=(n_sims, SIM_DAYS)
    strike_list,
    front_days_list,   # #days to expiry for front each strike
    back_days_list,    # #days to expiry for back  each strike
    front_iv_list,     # implied vol for front
    back_iv_list,      # implied vol for back
    net_debit_list,    # back_prem - front_prem
    is_call,
    r
):
    n_sims, sim_days = sim_px.shape
    n_strikes = len(strike_list)
    Z = np.zeros((sim_days, n_strikes))
    for i, K in enumerate(strike_list):
        fd = front_days_list[i]
        bd = back_days_list[i]
        fiv= front_iv_list[i]
        biv= back_iv_list[i]
        ndp= net_debit_list[i]
        for t in range(sim_days):
            # for each path => compute value
            vals = np.zeros(n_sims)
            for path_idx in range(n_sims):
                S_t = sim_px[path_idx, t]
                vals[path_idx] = value_calendar(
                    S=S_t,
                    K=K,
                    t=t,
                    front_exp_days=fd,
                    back_exp_days=bd,
                    front_iv=fiv,
                    back_iv=biv,
                    r=r,
                    net_debit=ndp,
                    is_call=is_call
                )
            Z[t,i] = vals.mean()
    return Z

##############################################################################
# MAIN
##############################################################################

def main():
    print("=== IonQ Calendar Spread (Advanced) ===")
    ib = IB()
    ib.connect(HOST, PORT, clientId=CLIENT_ID)
    print(f"Connected to IBKR at {HOST}:{PORT}")

    # 1) Load
    hist_df = load_historical_data()
    fund_df = load_fundamentals()
    macro_df= load_macro()
    merged_df = merge_data(hist_df, fund_df, macro_df)
    final_df = add_technical_features(merged_df)
    last_date= final_df.index[-1]  # last hist date
    print(f"Data up to {last_date}, shape={final_df.shape}")

    # 2) bubble + rfr
    shock_p, shock_m, shock_s = compute_shock_params(fund_df, macro_df)
    rfr = derive_rfr(macro_df)
    print(f"shock_p={shock_p:.3f}, shock_m={shock_m:.2f}, shock_s={shock_s:.2f}, rfr={rfr*100:.2f}%")

    # 3) GARCH => sim
    vol_path= fit_garch_volatility(final_df)
    df2 = final_df.copy()
    df2["log_ret"] = np.log(df2["close"]/df2["close"].shift(1))
    df2.dropna(inplace=True)
    mu= df2["log_ret"].mean()
    spot = final_df["close"].iloc[-1]
    print(f"Spot={spot:.2f}, mu={mu:.5f}")

    sim_prices= mc_sim(spot, mu, vol_path, shock_p, shock_m, shock_s)
    print(f"Sim shape={sim_prices.shape}")

    # 4) get chain => pick earliest 2 expiries after last_date
    stock = Stock('IONQ', 'SMART', 'USD')
    cont = ib.reqContractDetails(stock)[0].contract
    chain_params = ib.reqSecDefOptParams(cont.symbol, "", cont.secType, cont.conId)
    if not chain_params:
        print("No chain info. Exiting.")
        return
    chain_smart = None
    for cp in chain_params:
        if cp.exchange=="SMART":
            chain_smart=cp
            break
    if not chain_smart:
        print("No SMART chain. Exiting.")
        return

    # parse the expiries => pick the earliest 2 after last_date
    # last_date is e.g. 2024-12-31. we have chain_smart.expirations = {"20250124", ...}
    all_exps = sorted(chain_smart.expirations)
    # convert string yyyymmdd => datetime
    def parse_exp(s):
        return datetime.datetime.strptime(s, "%Y%m%d").date()
    after_last = [e for e in all_exps if parse_exp(e) > last_date.date()]
    if len(after_last)<2:
        print("We don't have at least 2 future expiries. Exiting.")
        return
    front_exp_str = after_last[0]
    back_exp_str  = after_last[1]
    front_exp_dt  = parse_exp(front_exp_str)
    back_exp_dt   = parse_exp(back_exp_str)
    print(f"Front expiry = {front_exp_str}, Back expiry= {back_exp_str}")

    # #days from last_date
    front_days = (front_exp_dt - last_date.date()).days
    back_days  = (back_exp_dt - last_date.date()).days
    print(f"Front days= {front_days}, Back days= {back_days}")

    # 5) filter near-money strikes
    min_strk= spot*(1.0 - STRIKE_RANGE)
    max_strk= spot*(1.0 + STRIKE_RANGE)
    candidate_strikes= sorted(s for s in chain_smart.strikes if min_strk<=s<=max_strk)

    # 6) For each strike => build front & back Option => fetch real impliedVol => compute net debit
    #    Also store front_days, back_days for each. We'll do "calls" or "puts" depending on config
    is_call = USE_CALLS_INSTEAD_OF_PUTS
    right_char = "C" if is_call else "P"

    def fetch_implied_vol(option_contract) -> float:
        """
        Request snapshot with genericTickList='100' => modelGreeks => impliedVol
        Must wait a bit.
        Return vol or nan
        """
        ib.reqMarketDataType(1)  # live
        tkr = ib.reqMktData(option_contract, "100,106", False, False)
        # wait for modelGreeks
        ib.sleep(SLEEP_FETCH)
        print(f"Ticker: {tkr}")
        if tkr.modelGreeks and tkr.modelGreeks.impliedVol is not None:
            vol = tkr.modelGreeks.impliedVol
            # clean up
            ib.cancelMktData(option_contract)
            return vol
        ib.cancelMktData(option_contract)
        return float('nan')

    def fetch_mid_price(option_contract) -> float:
        # fallback to standard approach if you want actual premium
        # but for a simpler approach, let's do the B/A from the model or tkr.bid/ask
        tkr = ib.reqMktData(option_contract, "", False, False)
        ib.sleep(SLEEP_FETCH)
        b,a = tkr.bid, tkr.ask
        ib.cancelMktData(option_contract)
        if (b is not None and b!=-1) and (a is not None and a!=-1):
            return 0.5*(b+a)
        if tkr.last is not None and tkr.last>0:
            return tkr.last
        if tkr.close is not None and tkr.close>0:
            return tkr.close
        return float('nan')

    results = []
    for K in candidate_strikes:
        # front
        c_front = Option("IONQ", front_exp_str, K, right_char, "SMART")
        # back
        c_back  = Option("IONQ", back_exp_str, K, right_char, "SMART")

        # qualify
        cdf = ib.reqContractDetails(c_front)
        cdb = ib.reqContractDetails(c_back)
        if not cdf or not cdb:
            continue

        # fetch implied vol
        f_iv = fetch_implied_vol(c_front)
        b_iv = fetch_implied_vol(c_back)
        if math.isnan(f_iv) or math.isnan(b_iv):
            continue

        # fetch mid price
        f_prem= fetch_mid_price(c_front)
        b_prem= fetch_mid_price(c_back)
        if math.isnan(f_prem) or math.isnan(b_prem):
            continue

        net_debit = b_prem - f_prem
        # must be >0 if we want a debit. If net_debit<0 => it's a credit, skip.
        # or if user only wants a net debit:
        if net_debit<0:
            continue

        results.append({
            "strike": K,
            "front_iv": f_iv,
            "back_iv" : b_iv,
            "front_prem": f_prem,
            "back_prem" : b_prem,
            "net_debit" : net_debit
        })

    if not results:
        print("No valid strikes that yield a net debit. Exiting.")
        return

    # build arrays
    results_df= pd.DataFrame(results).sort_values("strike")
    strike_list = results_df["strike"].values
    front_iv_list= results_df["front_iv"].values
    back_iv_list = results_df["back_iv"].values
    front_prem_list= results_df["front_prem"].values
    back_prem_list = results_df["back_prem"].values
    net_debit_list = results_df["net_debit"].values

    # 7) Build 3D array => day vs strike => average PnL
    Z = calendar_3d_surface(
        sim_px=sim_prices,
        strike_list=strike_list,
        front_days_list=[front_days]*len(strike_list),
        back_days_list =[back_days]*len(strike_list),
        front_iv_list= front_iv_list,
        back_iv_list = back_iv_list,
        net_debit_list= net_debit_list,
        is_call= is_call,
        r=rfr
    )

    # 8) For each strike => find the day that yields the max PnL
    #    Then pick the global best among all strikes
    best_global_pnl= -9999999.0
    best_global_day= 0
    best_global_strike= None
    for i,K in enumerate(strike_list):
        # PnL vs day => Z[:,i]
        arr = Z[:,i]
        local_max= arr.max()
        local_day= arr.argmax()
        if local_max>best_global_pnl:
            best_global_pnl= local_max
            best_global_day= local_day
            best_global_strike= K

    print(f"Auto-Selected Strike= {best_global_strike}, best PnL= {best_global_pnl:.2f} on day= {best_global_day}")

    # 9) 3D Plot => day vs strike => average PnL
    sim_days= Z.shape[0]
    n_strikes= Z.shape[1]
    x_axis= np.arange(sim_days)
    y_axis= strike_list
    X, Y= np.meshgrid(x_axis, y_axis, indexing='ij')  # shape=(days, n_strikes)

    fig= plt.figure(figsize=(10,6))
    ax= fig.add_subplot(111, projection='3d')
    surf= ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_title(f"IonQ {'Call' if is_call else 'Put'} Calendar Spread PnL Surface")
    ax.set_xlabel("Day Index")
    ax.set_ylabel("Strike")
    ax.set_zlabel("Avg PnL")
    plt.show()

    ib.disconnect()
    print("Disconnected. Done.")

if __name__=="__main__":
    main()
