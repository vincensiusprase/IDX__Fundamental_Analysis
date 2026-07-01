"""
Orkestrasi pipeline: proses satu ticker -> baris valuation & details,
lalu proses satu sektor (loop ticker + backtest + upload).

process_ticker() adalah main-loop lama yang dibungkus menjadi fungsi sehingga
logika perhitungan tetap identik dengan skrip asli.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from .mappings import (
    normalize_sector, normalize_industry,
    EV_SALES_TARGETS, EV_SALES_DEFAULT,
)
from .utils import get_item_safe, get_item_ttm, get_price_at_date, safe_div
from .valuation import (
    get_valid_methods, get_graham_constants,
    calc_graham_value, calc_dcf_value, calc_ev_ebitda_value,
    calc_weighted_fair_value, get_signal, fmt_cagr,
)
from .scoring import get_stock_type, calc_fundamental_score, score_label
from .backtest import run_backtest
from .sheets import upload_sheet, sort_and_clean


def process_ticker(ticker):
    """Proses satu emiten -> (rows_valuation, rows_details). Aman dari exception."""
    rows_val = []
    rows_det = []
    print(f"🔹 {ticker}...", end=" ")
    try:
        stock      = yf.Ticker(ticker)
        info       = stock.info
        raw_sector = info.get('sector', 'Unknown')
        raw_indust = info.get('industry', 'Unknown')

        sector   = normalize_sector(raw_sector)
        industry = normalize_industry(raw_indust)

        beta_raw = info.get('beta', 1.0) or 1.0
        beta     = float(np.clip(beta_raw, 0.3, 3.0))
        stock_type = get_stock_type(sector, industry)

        # Ambil semua data finansial
        is_ann = stock.financials
        bs_ann = stock.balance_sheet
        cf_ann = stock.cashflow
        is_q   = stock.quarterly_financials
        bs_q   = stock.quarterly_balance_sheet
        cf_q   = stock.quarterly_cashflow

        if is_ann.empty:
            print("❌ Tidak ada data."); return rows_val, rows_det

        hist = stock.history(period="10y")
        if hist.empty:
            print("❌ Tidak ada history harga."); return rows_val, rows_det

        shares_now = info.get('sharesOutstanding',
                     info.get('impliedSharesOutstanding', np.nan))

        # Helper lambdas
        def ann(k):  return get_item_safe(is_ann, k)
        def bsa(k):  return get_item_safe(bs_ann, k)
        def cfa(k):  return get_item_safe(cf_ann, k)

        net_income_a = ann(["Net Income", "NetIncome", "Net Income Common Stockholders"])
        revenue_a    = ann(["Total Revenue", "TotalRevenue", "Operating Revenue"])
        op_income_a  = ann(["Operating Income", "OperatingIncome", "EBIT"])
        gross_p_a    = ann(["Gross Profit", "GrossProfit"])
        int_exp_a    = ann(["Interest Expense", "InterestExpense"]).abs()
        da_a         = cfa(["Depreciation & Amortization",
                             "DepreciationAndAmortization", "Reconciled Depreciation"])
        ocf_a        = cfa(["Total Cash From Operating Activities", "Operating Cash Flow",
                             "Cash Flow From Continuing Operating Activities"])
        capex_a      = cfa(["Capital Expenditure", "Capital Expenditures"]).fillna(0)
        equity_a     = bsa(["Stockholders Equity", "StockholdersEquity",
                             "Total Equity Gross Minority Interest"])
        debt_a       = bsa(["Total Debt", "Long Term Debt And Capital Lease Obligation",
                             "Long Term Debt"])
        cash_a       = bsa(["Cash And Cash Equivalents",
                             "Cash Cash Equivalents And Short Term Investments"])
        curr_ass_a   = bsa(["Current Assets", "CurrentAssets"])
        curr_lia_a   = bsa(["Current Liabilities", "CurrentLiabilities"])
        inv_a        = bsa(["Inventory"])
        rec_a        = bsa(["Accounts Receivable", "Net Receivables", "Receivables"])
        tot_ass_a    = bsa(["Total Assets", "TotalAssets"])
        tot_lia_a    = bsa(["Total Liabilities Net Minority Interest", "Total Liabilities"])
        shares_a     = bsa(["Share Issued", "Ordinary Shares Number", "Common Stock"])
        shares_a     = shares_a.replace([0, np.nan], shares_now)
        div_a        = cfa(["Cash Dividends Paid", "CashDividendsPaid"]).abs()

        fcf_a = (ocf_a + capex_a)
        fcf_a.index = [d.year for d in fcf_a.index]
        fcf_a = fcf_a.sort_index()

        ebitda_a = op_income_a + da_a.fillna(0)
        valid_methods = get_valid_methods(sector)

        # ── PER TAHUN ────────────────────────────────────────────
        for col_date in list(is_ann.columns):
            yr = col_date.year
            try:
                price = get_price_at_date(hist, col_date)
                if pd.isna(price) or price <= 0: continue

                sh      = shares_a.get(col_date, shares_now) or shares_now
                rev     = revenue_a.get(col_date, np.nan)    or 0
                gp      = gross_p_a.get(col_date, np.nan)    or 0
                op      = op_income_a.get(col_date, np.nan)  or 0
                ni      = net_income_a.get(col_date, np.nan) or 0
                assets  = tot_ass_a.get(col_date, np.nan)    or 0
                eq      = equity_a.get(col_date, np.nan)     or 0
                lia     = tot_lia_a.get(col_date, np.nan)    or 0
                debt    = debt_a.get(col_date, np.nan)        or 0
                cash_v  = cash_a.get(col_date, np.nan)        or 0
                curr_a  = curr_ass_a.get(col_date, np.nan)   or 0
                curr_l  = curr_lia_a.get(col_date, np.nan)   or 0
                inv_v   = inv_a.get(col_date, np.nan)         or 0
                rec_v   = rec_a.get(col_date, np.nan)         or 0
                int_e   = int_exp_a.get(col_date, np.nan)    or 0
                da_v    = da_a.get(col_date, np.nan)          or 0
                div_v   = div_a.get(col_date, np.nan)         or 0
                ebitda  = ebitda_a.get(col_date, np.nan)      or 0

                eps  = safe_div(ni, sh, 0)
                bvps = safe_div(eq, sh, 0)

                # Rasio dasar
                gpm     = safe_div(gp,  rev,    0) * 100
                opm     = safe_div(op,  rev,    0) * 100
                npm     = safe_div(ni,  rev,    0) * 100
                roe     = safe_div(ni,  eq,     0) * 100
                roa     = safe_div(ni,  assets, 0) * 100
                der     = safe_div(lia, eq,     0)
                dar     = safe_div(lia, assets, 0)
                cr      = safe_div(curr_a, curr_l, 0)
                cogs_v  = (rev - gp) if gp else rev
                inv_to  = safe_div(cogs_v, inv_v, 0) if inv_v else np.nan
                rec_to  = safe_div(rev, rec_v, 0) if rec_v else np.nan
                at      = safe_div(rev, assets, 0)
                int_cov = safe_div(op, int_e, 0) if int_e else np.nan
                dps     = safe_div(div_v, sh, 0)
                div_yld = safe_div(dps, price, 0) * 100
                per     = safe_div(price, eps, 0) if eps > 0 else np.nan
                pbv     = safe_div(price, bvps, 0) if bvps > 0 else np.nan

                # ── Komponen baru ──────────────────────────────
                # FCF / Net Income (kualitas laba)
                ocf_v   = ocf_a.get(col_date, np.nan) or 0
                cpx_v   = capex_a.get(col_date, 0) or 0
                fcf_v   = ocf_v + cpx_v
                fcf_ni  = safe_div(fcf_v, ni, np.nan) if ni and ni > 0 else np.nan

                # Net Debt / EBITDA
                net_debt   = debt - cash_v
                nd_ebitda  = safe_div(net_debt, ebitda, np.nan) if ebitda and ebitda > 0 else np.nan

                # Revenue CAGR 3Y (%)
                rev_series = revenue_a.dropna()
                rev_series.index = [d.year for d in rev_series.index]
                rev_series = rev_series[rev_series > 0].sort_index()
                rev_base   = rev_series.get(yr - 3, np.nan)
                rev_cagr   = (safe_div(rev, rev_base) ** (1/3) - 1) * 100 if rev > 0 and not pd.isna(rev_base) and rev_base > 0 else np.nan

                # EPS CAGR 3Y (%)
                ni_series  = net_income_a.dropna()
                ni_series.index  = [d.year for d in ni_series.index]
                sh_series  = shares_a.dropna()
                sh_series.index  = [d.year for d in sh_series.index]
                eps_base_ni = ni_series.get(yr - 3, np.nan)
                eps_base_sh = sh_series.get(yr - 3, shares_now)
                eps_base   = safe_div(eps_base_ni, eps_base_sh, np.nan)
                eps_cagr   = (safe_div(eps, eps_base) ** (1/3) - 1) * 100 if eps > 0 and not pd.isna(eps_base) and eps_base > 0 else np.nan

                # CCC — Days Inventory + Days Receivable - Days Payable
                pay_a_col  = bs_ann
                payables_v = get_item_safe(pay_a_col, ["Accounts Payable","AccountsPayable","Payables"]).get(col_date, 0) or 0
                dio = safe_div(inv_v, cogs_v, np.nan) * 365 if inv_v and cogs_v else np.nan
                dso = safe_div(rec_v, rev, np.nan)    * 365 if rec_v and rev else np.nan
                dpo = safe_div(payables_v, cogs_v, np.nan) * 365 if payables_v and cogs_v else np.nan
                ccc = (dio + dso - dpo) if not any(pd.isna(x) for x in [dio, dso, dpo]) else np.nan

                # EV/EBITDA actual (recycle dari valuation)
                mkt_cap        = price * sh if price and sh else np.nan
                ev_actual      = (mkt_cap + debt - cash_v) if not pd.isna(mkt_cap) else np.nan
                ev_ebitda_act  = safe_div(ev_actual, ebitda, np.nan) if ebitda and ebitda > 0 else np.nan

                data_row = {
                    "GPM %":                 gpm,
                    "OPM %":                 opm,
                    "NPM %":                 npm,
                    "ROE %":                 roe,
                    "ROA %":                 roa,
                    "DER (x)":               der,
                    "DAR (x)":               dar,
                    "Interest_Coverage":     int_cov,
                    "Risk Factor":           np.nan,
                    "Asset Turnover (x)":    at,
                    "PER (x)":               per,
                    "PBV (x)":               pbv,
                    "Dividend Yield %":      div_yld,
                    "Current Ratio (x)":     cr,
                    "Inventory Turnover":    inv_to,
                    "Receivables TO":        rec_to,
                    # ── Baru ──
                    "FCF_to_NI":             fcf_ni,
                    "NetDebt_EBITDA":        nd_ebitda,
                    "Revenue_CAGR3":         rev_cagr,
                    "EPS_CAGR3":             eps_cagr,
                    "CCC":                   ccc,
                    "EV_EBITDA_score_input": ev_ebitda_act,
                }

                score, comp = calc_fundamental_score(data_row, stock_type, sector=sector)

                # Valuasi
                fcf_to_yr = fcf_a[fcf_a.index <= yr]
                g_val, g_k = calc_graham_value(eps, bvps, sector, industry) if valid_methods["graham"] else (np.nan, np.nan)
                d_val, d_wacc, d_cagr = calc_dcf_value(fcf_to_yr, sh, beta, debt, cash_v, revenue_total=rev, sector=sector) if valid_methods["dcf"] else (np.nan, np.nan, np.nan)
                ev_val, ev_mult = calc_ev_ebitda_value(ebitda, debt, cash_v, sh, sector) if valid_methods["ev_ebitda"] else (np.nan, np.nan)

                g_sig  = get_signal(price, g_val)  if not pd.isna(g_val)  else "N/A"
                d_sig  = get_signal(price, d_val)  if not pd.isna(d_val)  else "N/A"
                ev_sig = get_signal(price, ev_val) if not pd.isna(ev_val) else "N/A"
                # Label: DCF atau EV/Sales tergantung metode yang dipakai
                dcf_method_label = "EV/Sales" if isinstance(d_cagr, str) else "DCF"

                fair   = calc_weighted_fair_value(g_val, d_val, ev_val)
                upside = safe_div(fair - price, price) * 100 if not pd.isna(fair) else np.nan
                mos    = safe_div(fair - price, fair)  * 100 if not pd.isna(fair) and fair > 0 else np.nan

                # ── Append VALUATION ──
                rows_val.append({
                    "Ticker":               ticker,
                    "Sector (Yahoo)":       sector,
                    "Industry (Yahoo)":     industry,
                    "Stock_Type":           stock_type,
                    "Year":                 yr,
                    "Price":                round(price, 2),
                    "Beta":                 round(beta, 2),
                    "WACC (%)":             round(d_wacc * 100, 2) if not pd.isna(d_wacc) else np.nan,
                    "Graham_K":             round(g_k, 4) if not pd.isna(g_k) else np.nan,
                    "Graham_Value":         round(g_val, 2) if not pd.isna(g_val) else np.nan,
                    "Graham_Signal":        g_sig,
                    "DCF_Method":           dcf_method_label,
                    "DCF_FCF_CAGR (%)":     fmt_cagr(d_cagr) if dcf_method_label == "DCF" else np.nan,
                    "EV_Sales_Multiple":    EV_SALES_TARGETS.get(sector, EV_SALES_DEFAULT) if dcf_method_label == "EV/Sales" else np.nan,
                    "DCF_Value":            round(d_val, 2) if not pd.isna(d_val) else np.nan,
                    "DCF_Signal":           d_sig,
                    "EV_EBITDA_Multiple":   round(ev_mult, 1) if not pd.isna(ev_mult) else np.nan,
                    "EV_EBITDA_Value":      round(ev_val, 2) if not pd.isna(ev_val) else np.nan,
                    "EV_EBITDA_Signal":     ev_sig,
                    "Fair_Value":           round(fair, 2) if not pd.isna(fair) else np.nan,
                    "Upside_Pct (%)":       round(upside, 2) if not pd.isna(upside) else np.nan,
                    "Margin_of_Safety (%)": round(mos, 2) if not pd.isna(mos) else np.nan,
                    "Final_Signal":         get_signal(price, fair),
                    "Funda_Score":          round(score, 2) if not pd.isna(score) else np.nan,
                    "Funda_Label":          score_label(score),
                })

                # ── Append DETAILS ──
                detail_base = {
                    "Ticker": ticker, "Sector": sector, "Industry": industry,
                    "Stock_Type": stock_type, "Year": yr, "Price": round(price, 2),
                    "Funda_Score": round(score, 2) if not pd.isna(score) else np.nan,
                    "Funda_Label": score_label(score),
                }
                for lbl, c in comp.items():
                    raw = c["raw"]
                    detail_base[f"{lbl} | Raw"]       = round(raw, 4) if not pd.isna(raw) else np.nan
                    detail_base[f"{lbl} | Skor(1-5)"] = c["skor_1to5"]
                    detail_base[f"{lbl} | Bobot"]     = c["bobot"]
                    detail_base[f"{lbl} | Kontribusi"] = round(c["kontribusi"], 4)
                rows_det.append(detail_base)

            except Exception as e_inner:
                print(f"  [skip {yr}: {e_inner}]", end=" ")

        # ── TTM ──────────────────────────────────────────────────
        if not is_q.empty and not bs_q.empty:
            try:
                def qttm(k, t='flow'): return get_item_ttm(get_item_safe(is_q, k), t)
                def bttm(k):           return get_item_ttm(get_item_safe(bs_q, k), 'snapshot')
                def cfttm(k, t='flow'):return get_item_ttm(get_item_safe(cf_q, k), t)

                ni_t    = qttm(["Net Income", "NetIncome"]) or 0
                rev_t   = qttm(["Total Revenue", "TotalRevenue"]) or 0
                gp_t    = qttm(["Gross Profit"]) or 0
                op_t    = qttm(["Operating Income", "EBIT"]) or 0
                ocf_t   = cfttm(["Total Cash From Operating Activities", "Operating Cash Flow"]) or 0
                capx_t  = cfttm(["Capital Expenditure", "Capital Expenditures"]) or 0
                da_t    = cfttm(["Depreciation & Amortization", "DepreciationAndAmortization",
                                  "Reconciled Depreciation"]) or 0
                div_t   = cfttm(["Cash Dividends Paid", "CashDividendsPaid"]) or 0
                int_t   = abs(qttm(["Interest Expense", "InterestExpense"]) or 0)

                eq_t    = bttm(["Stockholders Equity", "StockholdersEquity"]) or 0
                debt_t  = bttm(["Total Debt", "Long Term Debt And Capital Lease Obligation"]) or 0
                cash_t  = bttm(["Cash And Cash Equivalents",
                                  "Cash Cash Equivalents And Short Term Investments"]) or 0
                ca_t    = bttm(["Current Assets", "CurrentAssets"]) or 0
                cl_t    = bttm(["Current Liabilities", "CurrentLiabilities"]) or 0
                inv_t   = bttm(["Inventory"]) or 0
                rec_t   = bttm(["Accounts Receivable", "Net Receivables"]) or 0
                ass_t   = bttm(["Total Assets", "TotalAssets"]) or 0
                lia_t   = bttm(["Total Liabilities Net Minority Interest",
                                  "Total Liabilities"]) or 0

                sh_t    = shares_now or 1
                price_t = hist.iloc[-1]['Close']

                # Risk Factor TTM (volatilitas 1 tahun)
                h2 = hist.copy()
                if h2.index.tz is not None: h2.index = h2.index.tz_localize(None)
                recent = h2[h2.index >= h2.index[-1] - pd.Timedelta(days=365)]['Close']
                rf_ttm = recent.pct_change().std() * np.sqrt(252) * 100 if len(recent) > 50 else np.nan

                eps_t   = safe_div(ni_t,  sh_t, 0)
                bvps_t  = safe_div(eq_t,  sh_t, 0)
                dps_t   = safe_div(abs(div_t), sh_t, 0)
                ebitda_t= op_t + da_t
                cogs_t  = (rev_t - gp_t) if gp_t else rev_t

                gpm_t    = safe_div(gp_t,  rev_t,  0) * 100
                opm_t    = safe_div(op_t,  rev_t,  0) * 100
                npm_t    = safe_div(ni_t,  rev_t,  0) * 100
                roe_t    = safe_div(ni_t,  eq_t,   0) * 100
                roa_t    = safe_div(ni_t,  ass_t,  0) * 100
                der_t    = safe_div(lia_t, eq_t,   0)
                dar_t    = safe_div(lia_t, ass_t,  0)
                cr_t     = safe_div(ca_t,  cl_t,   0)
                inv_to_t = safe_div(cogs_t, inv_t,  np.nan) if inv_t  else np.nan
                rec_to_t = safe_div(rev_t,  rec_t,  np.nan) if rec_t  else np.nan
                at_t     = safe_div(rev_t,  ass_t,  0)
                intcov_t = safe_div(op_t,   int_t,  np.nan) if int_t  else np.nan
                dyld_t   = safe_div(dps_t,  price_t, 0) * 100
                per_t    = safe_div(price_t, eps_t,  np.nan) if eps_t  > 0 else np.nan
                pbv_t    = safe_div(price_t, bvps_t, np.nan) if bvps_t > 0 else np.nan

                # ── Komponen baru TTM ──────────────────────────
                # FCF / Net Income
                fcf_ttm_v = ocf_t + capx_t
                fcf_ni_t  = safe_div(fcf_ttm_v, ni_t, np.nan) if ni_t and ni_t > 0 else np.nan

                # Net Debt / EBITDA
                net_debt_t  = debt_t - cash_t
                nd_ebitda_t = safe_div(net_debt_t, ebitda_t, np.nan) if ebitda_t and ebitda_t > 0 else np.nan

                # Revenue CAGR 3Y — bandingkan TTM vs revenue 3 tahun lalu
                rev_series_t = revenue_a.dropna()
                rev_series_t.index = [d.year for d in rev_series_t.index]
                rev_series_t = rev_series_t[rev_series_t > 0].sort_index()
                cur_yr   = datetime.now().year
                rev_3ago = rev_series_t.get(cur_yr - 3,
                           rev_series_t.get(cur_yr - 4, np.nan))
                rev_cagr_t = (safe_div(rev_t, rev_3ago) ** (1/3) - 1) * 100                              if rev_t > 0 and not pd.isna(rev_3ago) and rev_3ago > 0 else np.nan

                # EPS CAGR 3Y
                ni_ser_t  = net_income_a.dropna()
                ni_ser_t.index  = [d.year for d in ni_ser_t.index]
                sh_ser_t  = shares_a.dropna()
                sh_ser_t.index  = [d.year for d in sh_ser_t.index]
                eps_3ago_ni = ni_ser_t.get(cur_yr - 3, ni_ser_t.get(cur_yr - 4, np.nan))
                eps_3ago_sh = sh_ser_t.get(cur_yr - 3, sh_ser_t.get(cur_yr - 4, sh_t))
                eps_3ago    = safe_div(eps_3ago_ni, eps_3ago_sh, np.nan)
                eps_cagr_t  = (safe_div(eps_t, eps_3ago) ** (1/3) - 1) * 100                               if eps_t > 0 and not pd.isna(eps_3ago) and eps_3ago > 0 else np.nan

                # CCC TTM
                pay_t  = bttm(["Accounts Payable", "AccountsPayable", "Payables"]) or 0
                dio_t  = safe_div(inv_t,  cogs_t,  np.nan) * 365 if inv_t  and cogs_t  else np.nan
                dso_t  = safe_div(rec_t,  rev_t,   np.nan) * 365 if rec_t  and rev_t   else np.nan
                dpo_t  = safe_div(pay_t,  cogs_t,  np.nan) * 365 if pay_t  and cogs_t  else np.nan
                ccc_t  = (dio_t + dso_t - dpo_t)                          if not any(pd.isna(x) for x in [dio_t, dso_t, dpo_t]) else np.nan

                # EV/EBITDA TTM (recycle)
                mkt_cap_t     = price_t * sh_t if price_t and sh_t else np.nan
                ev_actual_t   = (mkt_cap_t + debt_t - cash_t) if not pd.isna(mkt_cap_t) else np.nan
                ev_ebitda_t_s = safe_div(ev_actual_t, ebitda_t, np.nan) if ebitda_t and ebitda_t > 0 else np.nan

                data_row_t = {
                    "GPM %":                 gpm_t,
                    "OPM %":                 opm_t,
                    "NPM %":                 npm_t,
                    "ROE %":                 roe_t,
                    "ROA %":                 roa_t,
                    "DER (x)":               der_t,
                    "DAR (x)":               dar_t,
                    "Interest_Coverage":     intcov_t,
                    "Risk Factor":           rf_ttm,
                    "Asset Turnover (x)":    at_t,
                    "PER (x)":               per_t,
                    "PBV (x)":               pbv_t,
                    "Dividend Yield %":      dyld_t,
                    "Current Ratio (x)":     cr_t,
                    "Inventory Turnover":    inv_to_t,
                    "Receivables TO":        rec_to_t,
                    # ── Baru ──
                    "FCF_to_NI":             fcf_ni_t,
                    "NetDebt_EBITDA":        nd_ebitda_t,
                    "Revenue_CAGR3":         rev_cagr_t,
                    "EPS_CAGR3":             eps_cagr_t,
                    "CCC":                   ccc_t,
                    "EV_EBITDA_score_input": ev_ebitda_t_s,
                }

                score_t, comp_t = calc_fundamental_score(data_row_t, stock_type, sector=sector)

                # FCF TTM series
                fcf_all_t = fcf_a.copy()
                fcf_ttm_val = ocf_t + capx_t
                if fcf_ttm_val > 0:
                    fcf_all_t[datetime.now().year] = fcf_ttm_val

                g_val_t, g_k_t = calc_graham_value(eps_t, bvps_t, sector, industry) if valid_methods["graham"] else (np.nan, np.nan)
                d_val_t, d_w_t, d_c_t = calc_dcf_value(fcf_all_t, sh_t, beta, debt_t, cash_t, revenue_total=rev_t, sector=sector) if valid_methods["dcf"] else (np.nan, np.nan, np.nan)
                ev_val_t, ev_m_t = calc_ev_ebitda_value(ebitda_t, debt_t, cash_t, sh_t, sector) if valid_methods["ev_ebitda"] else (np.nan, np.nan)

                fair_t  = calc_weighted_fair_value(g_val_t, d_val_t, ev_val_t)
                up_t    = safe_div(fair_t - price_t, price_t) * 100 if not pd.isna(fair_t) else np.nan
                mos_t   = safe_div(fair_t - price_t, fair_t)  * 100 if not pd.isna(fair_t) and fair_t > 0 else np.nan
                dcf_method_label_t = "EV/Sales" if isinstance(d_c_t, str) else "DCF"

                rows_val.append({
                    "Ticker":               ticker,
                    "Sector (Yahoo)":       sector,
                    "Industry (Yahoo)":     industry,
                    "Stock_Type":           stock_type,
                    "Year":                 "TTM",
                    "Price":                round(price_t, 2),
                    "Beta":                 round(beta, 2),
                    "WACC (%)":             round(d_w_t * 100, 2) if not pd.isna(d_w_t) else np.nan,
                    "Graham_K":             round(g_k_t, 4) if not pd.isna(g_k_t) else np.nan,
                    "Graham_Value":         round(g_val_t, 2) if not pd.isna(g_val_t) else np.nan,
                    "Graham_Signal":        get_signal(price_t, g_val_t),
                    "DCF_Method":           dcf_method_label_t,
                    "DCF_FCF_CAGR (%)":     fmt_cagr(d_c_t) if dcf_method_label_t == "DCF" else np.nan,
                    "EV_Sales_Multiple":    EV_SALES_TARGETS.get(sector, EV_SALES_DEFAULT) if dcf_method_label_t == "EV/Sales" else np.nan,
                    "DCF_Value":            round(d_val_t, 2) if not pd.isna(d_val_t) else np.nan,
                    "DCF_Signal":           get_signal(price_t, d_val_t),
                    "EV_EBITDA_Multiple":   round(ev_m_t, 1) if not pd.isna(ev_m_t) else np.nan,
                    "EV_EBITDA_Value":      round(ev_val_t, 2) if not pd.isna(ev_val_t) else np.nan,
                    "EV_EBITDA_Signal":     get_signal(price_t, ev_val_t),
                    "Fair_Value":           round(fair_t, 2) if not pd.isna(fair_t) else np.nan,
                    "Upside_Pct (%)":       round(up_t, 2) if not pd.isna(up_t) else np.nan,
                    "Margin_of_Safety (%)": round(mos_t, 2) if not pd.isna(mos_t) else np.nan,
                    "Final_Signal":         get_signal(price_t, fair_t),
                    "Funda_Score":          round(score_t, 2) if not pd.isna(score_t) else np.nan,
                    "Funda_Label":          score_label(score_t),
                })

                detail_ttm = {
                    "Ticker": ticker, "Sector": sector, "Industry": industry,
                    "Stock_Type": stock_type, "Year": "TTM", "Price": round(price_t, 2),
                    "Funda_Score": round(score_t, 2) if not pd.isna(score_t) else np.nan,
                    "Funda_Label": score_label(score_t),
                }
                for lbl, c in comp_t.items():
                    raw = c["raw"]
                    detail_ttm[f"{lbl} | Raw"]        = round(raw, 4) if not pd.isna(raw) else np.nan
                    detail_ttm[f"{lbl} | Skor(1-5)"]  = c["skor_1to5"]
                    detail_ttm[f"{lbl} | Bobot"]      = c["bobot"]
                    detail_ttm[f"{lbl} | Kontribusi"]  = round(c["kontribusi"], 4)
                rows_det.append(detail_ttm)

            except Exception as e_ttm:
                print(f"  [TTM err: {e_ttm}]", end=" ")

        print("✅")

    except Exception as e:
        print(f"❌ Error: {e}")
    return rows_val, rows_det


# ════════════════════════════════════════════════════════════════════════════
# ORKESTRASI PER SEKTOR
# ════════════════════════════════════════════════════════════════════════════
import time
from .config import YF_MAX_RETRIES, YF_RETRY_DELAY


def _fetch_history_with_retry(ticker, period="15y"):
    """Ambil history harga dengan retry sederhana untuk error koneksi transien."""
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            h = yf.Ticker(ticker).history(period=period)
            return h if not h.empty else None
        except Exception as e:
            if attempt == YF_MAX_RETRIES:
                print(f"   ⚠️  history {ticker} gagal ({e})")
                return None
            time.sleep(YF_RETRY_DELAY)
    return None


def run_sector(code, cfg, sh):
    """
    Proses satu sektor penuh: loop ticker -> valuasi -> backtest -> upload.
    Return dict ringkasan. Exception per-sektor tidak dilempar ke atas.
    """
    tickers = cfg["tickers"]
    print(f"\n{'='*60}\n🏭 SEKTOR {code} — {len(tickers)} saham\n{'='*60}")

    all_valuation, all_details = [], []
    for ticker in tickers:
        rows_val, rows_det = process_ticker(ticker)
        all_valuation.extend(rows_val)
        all_details.extend(rows_det)

    if not all_valuation:
        print(f"❌ {code}: tidak ada data berhasil diproses.")
        return {"sector": code, "rows": 0, "status": "EMPTY"}

    df_val = sort_and_clean(pd.DataFrame(all_valuation))
    df_det = sort_and_clean(pd.DataFrame(all_details))

    # ── Backtest enrichment ──────────────────────────────────────────────
    print(f"\n📈 Backtest {code} (ambil history per ticker)...")
    hist_cache = {}
    for i, t in enumerate(df_val["Ticker"].unique(), 1):
        h = _fetch_history_with_retry(t)
        if h is not None:
            hist_cache[t] = h
        print(f"   [{i}] {t}", end="\r")
    df_bt = run_backtest(df_val, hist_cache)
    df_val = pd.concat([df_val, df_bt], axis=1)

    # ── Upload ───────────────────────────────────────────────────────────
    print(f"\n📤 Upload {code} ke Google Sheets...")
    upload_sheet(sh, df_val, cfg["sheet_valuation"], cols=60)
    upload_sheet(sh, df_det, cfg["sheet_details"])
    print(f"✅ {code} selesai — {len(df_val)} baris valuation")
    return {"sector": code, "rows": len(df_val), "status": "OK"}
