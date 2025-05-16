import pandas as pd
import time
import json
import requests
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)

class NSEMasterData:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1',
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            'Content-Type': 'application/json',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate'
        })
        self.nse_url = "https://charting.nseindia.com/Charts/GetEQMasters"
        self.nfo_url = "https://charting.nseindia.com/Charts/GetFOMasters"
        self.historical_url = "https://charting.nseindia.com//Charts/symbolhistoricaldata/"
        self.nse_data = None
        self.nfo_data = None

    def search(self, symbol, exchange, match=False):
        """Search for symbols in the specified exchange."""
        exchange = exchange.upper()
        if exchange == 'NSE':
            df = self.nse_data
        elif exchange == 'NFO':
            df = self.nfo_data
        else:
            logger.error(f"Invalid exchange '{exchange}'. Please choose 'NSE' or 'NFO'.")
            return pd.DataFrame()

        if df is None:
            logger.error(f"Data for {exchange} not downloaded. Please run download() first.")
            return pd.DataFrame()

        if match:
            result = df[df['Symbol'].str.upper() == symbol.upper()]
        else:
            result = df[df['Symbol'].str.contains(symbol, case=False, na=False)]

        if result.empty:
            logger.warning(f"No matching result found for symbol '{symbol}' in {exchange}.")
            return pd.DataFrame()

        return result.reset_index(drop=True)

    def get_nse_symbol_master(self, url):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.text.splitlines()
            columns = ['ScripCode', 'Symbol', 'Name', 'Type']
            return pd.DataFrame([line.split('|') for line in data], columns=columns)
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data from {url}: {e}")
            return pd.DataFrame()

    def download_symbol_master(self):
        """Download NSE and NFO master data."""
        self.nse_data = self.get_nse_symbol_master(self.nse_url)
        self.nfo_data = self.get_nse_symbol_master(self.nfo_url)

    def search_symbol(self, symbol, exchange):
        """Search for a symbol in the specified exchange and return the first match."""
        df = self.nse_data if exchange.upper() == 'NSE' else self.nfo_data
        if df is None:
            logger.error(f"Data for {exchange} not downloaded. Please run download() first.")
            return None
        result = df[df['Symbol'].str.contains(symbol, case=False, na=False)]
        if result.empty:
            logger.warning(f"No matching result found for symbol '{symbol}' in {exchange}.")
            return None
        return result.iloc[0]

    def get_history(self, symbol="Nifty 50", exchange="NSE", start=None, end=None, interval='1d'):
        """Get historical data for a symbol."""
        def adjust_timestamp(ts):
            if interval in ['30m', '1h']:
                num = 15
            elif interval in ['10m']:
                num = 5
            else:
                num = int(re.match(r'\d+', interval).group())
            if num == 0:
                return (ts - timedelta(minutes=num)).round('min')
            else:
                return (ts - timedelta(minutes=num)).round((str(num) + 'min'))

        symbol_info = self.search_symbol(symbol, exchange)
        if symbol_info is None:
            return pd.DataFrame()

        interval_xref = {
            '1m': ('1', 'I'), '3m': ('3', 'I'), '5m': ('5', 'I'), '10m': ('5', 'I'),
            '15m': ('15', 'I'), '30m': ('15', 'I'), '1h': ('15', 'I'),
            '1d': ('1', 'D'), '1w': ('1', 'W'), '1M': ('1', 'M')
        }

        time_interval, chart_period = interval_xref.get(interval, ('1', 'D'))

        payload = {
            "exch": "N" if exchange.upper() == "NSE" else "D",
            "instrType": "C" if exchange.upper() == "NSE" else "D",
            "ScripCode": int(symbol_info['ScripCode']),
            "ulScripCode": int(symbol_info['ScripCode']),
            "fromDate": int(start.timestamp()) if start else 0,
            "toDate": int(end.timestamp()) if end else int(time.time()),
            "timeInterval": time_interval,
            "chartPeriod": chart_period,
            "chartStart": 0
        }

        try:
            # Set Cookies
            self.session.get("https://www.nseindia.com", timeout=5)
            response = self.session.post(self.historical_url, data=json.dumps(payload), timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                logger.warning("No data received from the Source - NSE.")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            df.columns = ['Status', 'TS', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['TS'] = pd.to_datetime(df['TS'], unit='s', utc=True)
            df['TS'] = df['TS'].dt.tz_localize(None)
            df = df[['TS', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Apply cutoff time only for intraday intervals
            intraday_intervals = ['1m', '3m', '5m', '15m']
            intraday_consolidate_intervals = ['10m','30m', '1h']
            if interval in intraday_intervals:
                cutoff_time = pd.Timestamp('15:30:00').time()
                df = df[df['TS'].dt.time <= cutoff_time]
                df['Timestamp'] = df['TS'].apply(adjust_timestamp)
                df.drop(columns=['TS'], inplace=True)
                df.set_index('Timestamp', inplace=True, drop=True)
                return df
            if interval in intraday_consolidate_intervals:
                cutoff_time = pd.Timestamp('15:30:00').time()
                df = df[df['TS'].dt.time <= cutoff_time]
                df['Timestamp'] = df['TS'].apply(adjust_timestamp)
                df.drop(columns=['TS'], inplace=True)
                df.set_index('Timestamp', inplace=True, drop=True)
                agg_parm = ''
                if interval == '30m':
                    agg_parm = '30min'
                elif interval == '10m':
                    agg_parm = '10min'
                else:
                    agg_parm = '60min'
                # Get the first timestamp to use as custom origin
                first_ts = df.index.min()
                offset_td = pd.to_timedelta(first_ts.time().strftime('%H:%M:%S'))
                df_aggregated = df.resample(agg_parm, origin='start_day', offset=offset_td).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                df_aggregated.dropna(inplace=True)
                return df_aggregated

            df.rename(columns={'TS': 'Timestamp'}, inplace=True)
            df.set_index('Timestamp', inplace=True, drop=True)
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"An error occurred while fetching historical data: {e}")
            return pd.DataFrame() 