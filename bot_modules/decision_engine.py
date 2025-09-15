import pandas as pd

def find_trade_signals(analysed_df: pd.DataFrame, confidence_threshold: float = 0.90) -> pd.DataFrame:
    """
    Filters the raw AI analysis to find credible trade signals.
    A good signal requires both strong sentiment (not neutral) and high confidence
    from the model in its own prediction.
    """
    # This is a filtering operation. We select rows from the DataFrame
    # where both of the following conditions are true (the '&' means AND):
    # 1. The 'sentiment' column is not equal to 'neutral'.
    # 2. The 'confidence' column is greater than or equal to our threshold.
    signals = analysed_df[
        (analysed_df['sentiment'] != 'neutral') &
        (analysed_df['confidence'] >= confidence_threshold)
    ].copy()

    # We add a new 'signal' column to make the trading action explicit.
    # The .apply() method runs a function on every row of the 'sentiment' column.
    # The 'lambda' function is a small, anonymous function that checks the value:
    # if the sentiment is 'positive', it returns 'BUY', otherwise it returns 'SELL'.
    signals['signal'] = signals['sentiment'].apply(lambda x: 'BUY' if x == 'positive' else 'SELL')
    
    return signals

def rank_signals(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks the credible trade signals to find the best opportunity.
    If the bot finds multiple signals in a single run, it needs a logical
    way to decide which one to act on.
    """
    # This ranks the signals by sorting the DataFrame.
    # `by='confidence'` tells pandas which column to sort by.
    # `ascending=False` ensures that the highest confidence score is at the top.
    ranked_signals = signals_df.sort_values(by='confidence', ascending=False)
    
    # We reset the index for a clean, 0-based sequence after sorting.
    ranked_signals.reset_index(drop=True, inplace=True)
    return ranked_signals

def check_ma_crossover_signal(price_history: pd.DataFrame, fast_window: int = 20, slow_window: int = 50) -> str:
    """
    Provides a technical confirmation signal based on market trend.
    This uses the Dual Moving Average Crossover strategy from an earlier project on my GitHub.
    The function returns 'BUY', 'SELL', or 'HOLD' based on the crossover of two SMAs.
    """
    # Copy to avoid modifying the original DataFrame.
    price_history = price_history.copy()
    
    # A Simple Moving Average (SMA) is the average closing price over a number of days (the "window").
    # The .rolling() method creates a rolling window, and .mean() calculates the average inside it.
    price_history['fast_sma'] = price_history['close'].rolling(window=fast_window).mean()
    price_history['slow_sma'] = price_history['close'].rolling(window=slow_window).mean()

    # We only care about the most recent value to know the current trend.
    # .iloc[-1] is a pandas command to get the very last item in a Series.
    last_fast_sma = price_history['fast_sma'].iloc[-1]
    last_slow_sma = price_history['slow_sma'].iloc[-1]

    # If the fast-moving average (recent trend) is above the slow one (long-term trend),
    # it's a bullish sign.
    if last_fast_sma > last_slow_sma:
        return 'BUY'
    # If the fast-moving average is below the slow one, it's a bearish sign.
    elif last_fast_sma < last_slow_sma:
        return 'SELL'
    # Otherwise, the trend is unclear or there's not enough data to calculate the SMAs.
    else:
        return 'HOLD'