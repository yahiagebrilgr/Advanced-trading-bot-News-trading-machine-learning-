import os
import pandas as pd
from dotenv import load_dotenv

# The Alpaca library is split into a client for trading and a client for data.
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

# These are helper classes and enums (enumerations) from the Alpaca library.
# Enums are used to make code safer and more readable. Instead of typing a string
# like "buy", we can use `OrderSide.BUY`, which prevents spelling mistakes.

from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.timeframe import TimeFrame

def get_broker_api() -> TradingClient:
    """
    Establishes a secure connection to the Alpaca trading API using the
    credentials stored in the .env file.
    """
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    if not api_key or not api_secret:
        raise ValueError("API_KEY and API_SECRET must be set in the .env file.")
    
    # The 'paper=True' argument is vital. It tells the client to connect
    # to the paper trading (simulation) account, not a live money account. 
    # I dont enough confidence in this bot to run my real money yet :)
    return TradingClient(api_key, api_secret, paper=True)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculates the Average True Range (ATR).
    ATR is a key indicator of market volatility. We use it to set dynamic
    stop-loss and take-profit levels that adapt to how much a stock is
    currently moving. A volatile stock gets wider targets, whereas a calm stock gets tighter ones.
    """
    df = df.copy()
    # The True Range is the greatest of the following:
    # 1. The current day's high minus the current day's low.
    df['high-low'] = df['high'] - df['low']
    # 2. The absolute value of the current day's high minus the previous day's close.
    df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
    # 3. The absolute value of the current day's low minus the previous day's close.
    df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
    
    df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
    
    # The ATR is a smoothed moving average of the True Range.
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    return df

def execute_trade_signal(api: TradingClient, signal_data: pd.Series, cash_to_use: float):
    """
    Takes a single trade signal and executes a complete, risk-managed
    bracket order via the broker API.
    """
    symbol = signal_data['tickers'][0]
    signal = signal_data['signal']
    
    print(f"\nPreparing to execute trade for {symbol}")

    try:
        # Get live data for planning the trade
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        data_client = StockHistoricalDataClient(api_key, api_secret)
        
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            limit=100  # We need enough data to calculate the ATR accurately.
        )
        bars = data_client.get_stock_bars(request_params).df
        
        # We need to ensure the column names are lowercase for our ATR function.
        bars.columns = bars.columns.str.lower()
        bars = calculate_atr(bars, period=14)
        last_price = bars['close'].iloc[-1]
        atr = bars['atr'].iloc[-1]
        
        # Below section is for position sizing 
        # We calculate how many shares we can buy/sell with the cash allocated by the live bot.
        quantity = round(cash_to_use / last_price)
        print(f"  - Position Size: {quantity} shares of {symbol} at ~${last_price:.2f}")

        # We then define the bracket order
        # A bracket order is a three-part order that automates our risk management.
        if signal == 'BUY':
            order_details = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC, # GTC = "Good 'til Canceled"
                order_class=OrderClass.BRACKET,
                # Take-Profit: Set at 4x the ATR above our entry price.
                take_profit=TakeProfitRequest(limit_price=round(last_price + (atr * 4), 2)),
                # Stop-Loss: Set at 2x the ATR below our entry price.
                stop_loss=StopLossRequest(stop_price=round(last_price - (atr * 2), 2))
            )
        elif signal == 'SELL':
            order_details = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(last_price - (atr * 4), 2)),
                stop_loss=StopLossRequest(stop_price=round(last_price + (atr * 2), 2))
            )
        else:
            print(f"Invalid signal '{signal}'. No order placed.")
            return

        # Section belwo submitsthe order
        print("Submitting bracket order")
        order = api.submit_order(order_data=order_details)
        print(f"  - Order submitted successfully. Order ID: {order.id}")

    except Exception as e:
        print(f"Failed to execute trade for {symbol}. Error: {e}")