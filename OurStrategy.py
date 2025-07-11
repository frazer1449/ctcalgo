
import pandas as pd
import numpy as np
import math
import random
from datetime import datetime, timedelta, time
from scipy.stats import norm
from scipy.optimize import brentq, fmin
import traceback


class Strategy:
    def __init__(self, start, end, options_data, underlying):
        # Set starting capital
        self.capital = 100_000_000
        self.portfolio_value = 0
        self.pnl = []  # Track PnL for each order

        # Set start and end dates of the given timeframe
        self.start_date = start #datetime.strptime(start, '%Y-%m-%d') 
        self.end_date = end # datetime.strptime(end, '%Y-%m-%d')

        # Load options data
        self.options = pd.read_csv(options_data)
        self.options["day"] = self.options["ts_recv"].apply(lambda x: str(x).split("T")[0])
        self.options['hourmin'] = self.options["ts_recv"].apply(
            lambda x: datetime.strptime(x.split("T")[1][:8], "%H:%M:%S").time())
        
        self.options["strikeDate"] = "20" + self.options["symbol"].str[6:8] + "-" + self.options["symbol"].str[8:10] + "-" + self.options["symbol"].str[10:12]
        self.options['strikeDate'] = pd.to_datetime(self.options['strikeDate'])
        self.options = self.options[self.options['strikeDate'] < self.end_date]
        self.options["price"] = pd.to_numeric(self.options["symbol"].str[13:18])
        self.options["mid"] = (self.options['bid_px_00'] + self.options['ask_px_00']) / 2

        # Load underlying data
        self.underlying = pd.read_csv(underlying)
        self.underlying.columns = self.underlying.columns.str.lower()

        # mx_of_day price date
        # 34260000 4742.96 20240102
        self.underlying['date'] = self.underlying['date'].astype(str)
        self.underlying['day'] = self.underlying['date'].apply(lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:8])
        self.underlying['hourmin'] = self.underlying['ms_of_day'].apply(
            lambda x: str(self.convert_ms_to_hhmm(x)[0]) + ":" + str(self.convert_ms_to_hhmm(x)[1])
        )
        self.underlying['hourmin'] = self.underlying['hourmin'].apply(
            lambda x: self.convert_military_time(x)
        )
        self.underlying = self.underlying[["price","day","hourmin"]]
        self.underlying = self.underlying[self.underlying["hourmin"].astype(str) != "21:00:00"].reset_index(drop=True)

    def convert_ms_to_hhmm(self, milliseconds):
        total_seconds = milliseconds // 1000
        total_minutes = total_seconds // 60
        hours = total_minutes // 60
        remaining_minutes = total_minutes % 60
        return [hours + 5, remaining_minutes]
    
    def convert_military_time(self, military_time_str):
      hours, minutes = military_time_str.split(':')
      
      # Pad the minutes with a leading zero if it's a single digit
      if len(minutes) == 1:
          minutes = '0' + minutes
      
      # Create a time object from the corrected hours and minutes
      hourmin_time = time(int(hours), int(minutes))
      # Combine with today's date to create a datetime object
      # combined_datetime = datetime.combine(datetime.today(), hourmin_time)
      return hourmin_time
    def generate_orders(self) -> pd.DataFrame:
        orders = []

        underlying = self.underlying
        options = self.options
        for i in range(0, len(underlying),30):
            chunk = underlying.iloc[i:i+30].reset_index(drop=True)
            mean_price_30min = chunk['price'].mean()
            try:
                day = str(chunk['day'].iloc[0])
                hourmin_start = chunk['hourmin'].iloc[0]
                hourmin_end = chunk['hourmin'].iloc[29]
                valid_options = options[
                    (options["day"] == day) &
                    (options['hourmin'] >= hourmin_start) &
                    (options['hourmin'] <= hourmin_end)
                    ].copy()
                if valid_options.empty:
                    # print(f"Empty for day: {day}, hourmin_start: {hourmin_start}")
                    continue
                current_price = (math.floor(mean_price_30min / 5 + 0.5)) * 5
                
                valid_options['curDate'] = datetime.strptime(day, '%Y-%m-%d')
                valid_options['dayDiff'] = valid_options['strikeDate'] - valid_options['curDate']
                valid_options = valid_options[valid_options['dayDiff'].dt.days<93]
                # valid_options = valid_options[(valid_options['price']<current_price+500)&(valid_options['price']>current_price-500)]
                
                valid_options['implied_vol'] = valid_options.apply(CalcImpliedVol, axis=1, args=(current_price, 0.03))
                valid_options_filtered = valid_options[(valid_options['implied_vol'] > 0)]
                
                if valid_options_filtered.empty:
                    continue

                df = valid_options_filtered.sort_values(['strikeDate']).drop_duplicates('strikeDate')
                strike_date = df.iloc[1]['strikeDate']

                vol = valid_options_filtered['implied_vol'].mean()

                str_strike_date = str(strike_date)[:10]

                order_df = order2(current_price, vol, str_strike_date, valid_options_filtered)
                print("order_df: ", order_df)
                if not order_df.empty:
                    orders.append(order_df)
                # print(len(orders))
            except:
                pass
        all_orders = pd.concat(orders, ignore_index=True)
        all_orders.rename(columns={'ts_recv': 'datetime'}, inplace=True)
        return all_orders

    def calculate_pnl(self, order, entry_price, exit_price):
        """
        Simulates the PnL for a given order.
        The current strategy assumes options expire at strike date, so the profit/loss is calculated
        based on the difference between the entry and exit price.
        """
        # Calculate PnL
        if order["action"] == "B":
            pnl = (exit_price - entry_price) * order["order_size"]
        else:
            pnl = (entry_price - exit_price) * order["order_size"]

        self.portfolio_value += pnl
        return pnl


# Helper functions
def CalcImpliedVol(row, S, r):
    if row['symbol'][12] == "C":
        return CalcImpliedVolCall(S, row['price'], row['dayDiff'].days / 365, r, row['mid'])
    else:
        return CalcImpliedVolPut(S, row['price'], row['dayDiff'].days / 365, r, row['mid'])


def CalcImpliedVolCall(S, K, T, r, mp):
    def bs_price(sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - (sigma * np.sqrt(T))
        BSprice_call = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        return BSprice_call - mp
    try:
        return brentq(bs_price, 0.0001, 100, maxiter=100)
    except Exception:
        pass
    
def CalcImpliedVolPut(S, K, T, r, mp):
    def implied_volatility_put(s):
        d1 = (np.log(S / K) + (r + 0.5 * s[0] ** 2) * T) / (s[0] * np.sqrt(T))
        d2 = d1 - s[0] * np.sqrt(T)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return (put_price - mp) ** 2

    return fmin(implied_volatility_put, [0.12], disp=False)[0]

# Order generation based on volatility strategy
def find_option(df, strike_date, strike_price, option_type):
    """
    Helper function to find the option row with the given strike date, strike price, and option type (C/P).
    """
    option_row = df[
        (df['strikeDate'] == strike_date) &    # Match the strike date
        (df['price'] == strike_price) &        # Match the strike price
        (df['symbol'].str.contains(option_type))  # Match 'C' for calls or 'P' for puts
    ]
    i=1
    while option_row.empty and i<=1: 
        option_row = df[
            (df['strikeDate'] == strike_date) &    # Match the strike date
            (df['price'] <= strike_price+i) &        # Match the strike price
            (df['price'] >= strike_price-i) &
            (df['symbol'].str.contains(option_type))  # Match 'C' for calls or 'P' for puts
        ]
        i=i+1
    if not option_row.empty:
        return option_row.iloc[0]
    else:
        return None
    

def order2(cPrice, vol, strikeDate, df):
    cPrice = (math.floor(cPrice / 5 + 0.5)) * 5  # Normalize current price
    order_list = []

    if vol < 0.05:
        # Iron Butterfly Strategy
        call_buy = find_option(df, strikeDate, cPrice + 30, 'C')
        call_sell = find_option(df, strikeDate, cPrice, 'C')
        put_sell = find_option(df, strikeDate, cPrice, 'P')
        put_buy = find_option(df, strikeDate, cPrice - 30, 'P')

        if call_buy is not None and call_sell is not None and put_sell is not None and put_buy is not None:
            order_list.append([call_buy['ts_recv'], call_buy['symbol'], "B", 10])
            order_list.append([call_sell['ts_recv'], call_sell['symbol'], "S", 10])
            order_list.append([put_sell['ts_recv'], put_sell['symbol'], "S", 10])
            order_list.append([put_buy['ts_recv'], put_buy['symbol'], "B", 10])

    elif vol < 0.1:
        # Iron Condor Strategy
        call_buy = find_option(df, strikeDate, cPrice + 60, 'C')
        call_sell = find_option(df, strikeDate, cPrice + 30, 'C')
        put_sell = find_option(df, strikeDate, cPrice - 30, 'P')
        put_buy = find_option(df, strikeDate, cPrice - 60, 'P')

        if call_buy is not None and call_sell is not None and put_sell is not None and put_buy is not None:
            order_list.append([call_buy['ts_recv'], call_buy['symbol'], "B", 10])
            order_list.append([call_sell['ts_recv'], call_sell['symbol'], "S", 10])
            order_list.append([put_sell['ts_recv'], put_sell['symbol'], "S", 10])
            order_list.append([put_buy['ts_recv'], put_buy['symbol'], "B", 10])

    elif vol < 0.2:
        # Long Straddle Strategy
        call_buy = find_option(df, strikeDate, cPrice, 'C')
        put_buy = find_option(df, strikeDate, cPrice, 'P')

        if call_buy is not None and put_buy is not None:
            order_list.append([call_buy['ts_recv'], call_buy['symbol'], "B", 10])
            order_list.append([put_buy['ts_recv'], put_buy['symbol'], "B", 10])

    else:
        # Long Strangle Strategy
        call_buy = find_option(df, strikeDate, cPrice + 30, 'C')
        put_buy = find_option(df, strikeDate, cPrice - 30, 'P')

        if call_buy is not None and put_buy is not None:
            order_list.append([call_buy['ts_recv'], call_buy['symbol'], "B", 10])
            order_list.append([put_buy['ts_recv'], put_buy['symbol'], "B", 10])

    # Convert the order list to a DataFrame
    if order_list:
        order_df = pd.DataFrame(order_list, columns=["ts_recv", "option_symbol", "action", "order_size"])
        return order_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no options found

# Helper function to build option symbols
def build_option_symbol(strikeDate, putCall, strikePrice):
    strike_date = strikeDate[2:4] + strikeDate[5:7] + strikeDate[8:10]
    strike_price = str(strikePrice)
    strike_price = '0' * (5 - len(strike_price)) + strike_price
    symbol = 'SPX ' + strike_date + putCall + strike_price + "000"
    return symbol

# New helper function that builds symbol WITHOUT spaces for output purposes
def build_option_symbol_output(strikeDate, putCall, strikePrice):
    strike_date = strikeDate[2:4] + strikeDate[5:7] + strikeDate[8:10]
    strike_price = str(strikePrice)
    strike_price = '0' * (5 - len(strike_price)) + strike_price
    symbol = 'SPX ' + strike_date + putCall + strike_price + "000"
    return symbol

# s = Strategy('2024-01-01','2024-03-30',"data/cleaned_options_data.csv","data/spx_minute_level_data_jan_mar_2024.csv")
# print(s.underlying[360:391])
# print(s.underlying.dtypes)

# df = s.underlying
# options = s.options
# for i in range(0, len(df),30):
#     chunk = df.iloc[i:i+30].reset_index(drop=True)

# print(options.head())
# print(len(options))
# for i in range(0, len(df),30):
#     chunk = df.iloc[i:i+30].reset_index(drop=True)
#     mean_price_30min = chunk['price'].mean()
#     day = str(chunk['day'].iloc[0])
#     hourmin_start = chunk['hourmin'].iloc[0]
#     hourmin_end = chunk['hourmin'].iloc[29]
#     valid_options = options[
#         (options["day"] == day) &
#         (options['hourmin'] >= hourmin_start) &
#         (options['hourmin'] <= hourmin_end)
#         ].copy()
    # print(i, len(valid_options))
# print(df[df['day']=='2024-01-03'])
# print(len(df[df['day']=='2024-01-04']))
# print(len(df[df['day']=='2024-01-05']))
# print(len(df[df['day']=='2024-01-06']))
# print(len(df[df['day']=='2024-01-07']))
# print(len(df[df['day']=='2024-01-08']))
# print(len(df[df['day']=='2024-01-09']))
# print(len(df[df['day']=='2024-01-10']))
# print(len(df[df['day']=='2024-01-11']))
# print(len(df[df['day']=='2024-01-12']))
# print(len(df[df['day']=='2024-01-13']))
# print(len(df[df['day']=='2024-01-14']))
# print(len(df[df['day']=='2024-01-15']))
# print(len(df[df['day']=='2024-01-16']))
# print(len(df[df['day']=='2024-01-17']))
# print(len(df[df['day']=='2024-01-18']))
# print(len(df[df['day']=='2024-01-19']))
# print(len(df[df['day']=='2024-01-20']))
# print(len(df[df['day']=='2024-01-21']))
# print(len(df[df['day']=='2024-01-22']))
# print(len(df[df['day']=='2024-01-23']))



    