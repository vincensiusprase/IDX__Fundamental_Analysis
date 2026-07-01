"""Helper murni untuk ekstraksi & pembersihan data numerik."""
import numpy as np
import pandas as pd


def get_item_safe(df, keywords):
    for k in keywords:
        if k in df.index:
            return df.loc[k]
    return pd.Series([np.nan] * len(df.columns), index=df.columns)

def get_item_ttm(series_data, item_type='flow'):
    try:
        if isinstance(series_data, (int, float, np.number)):
            return series_data
        if len(series_data) == 0 or series_data.isna().all():
            return np.nan
        return series_data.iloc[:4].sum() if item_type == 'flow' else series_data.iloc[0]
    except:
        return np.nan

def get_price_at_date(history_df, target_date):
    try:
        h = history_df.copy()
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        if hasattr(target_date, 'tz_localize'):
            target_date = target_date.tz_localize(None)
        h = h.sort_index()
        idx = h.index.get_indexer([target_date], method='pad')[0]
        return np.nan if idx == -1 else h.iloc[idx]['Close']
    except:
        return np.nan

def safe_div(a, b, default=np.nan):
    try:
        if b == 0 or pd.isna(b) or np.isinf(b):
            return default
        r = a / b
        return r if not (pd.isna(r) or np.isinf(r)) else default
    except:
        return default

