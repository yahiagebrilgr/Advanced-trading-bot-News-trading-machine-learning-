import pandas as pd
import ast # Used to safely convert string representations of lists back into lists.
import matplotlib.pyplot as plt # library for creating charts and graphs, from CodeSignal course. 

# We import all the functions from our bot's decision engine and analyser modules. 
from bot_modules.decision_engine import find_trade_signals, rank_signals, check_ma_crossover_signal
from bot_modules.analyser import analyse_sentiment_of_headlines, setup_finbert_model

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    # Calculates the Average True Range (ATR) for risk management
    df = df.copy()
    df['high-low'] = df['high'] - df['low']
    df['high-prev_close'] = abs(df['high'] - df['close'].shift(1))
    df['low-prev_close'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high-low', 'high-prev_close', 'low-prev_close']].max(axis=1)
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    return df

def run_backtest():
    """
    Runs the full backtesting simulation, kind of like a flight simulator for the trading strategy.
    """
    print("Starting Backtest with Technical Filter")

    # 1. Load the Historical Timelines
    # We load our two key historical datasets: one for news events and one for prices.
    try:
        news_df = pd.read_csv('data/historical_news_dataset.csv', parse_dates=['timestamp'])
        price_df = pd.read_csv('data/historical_price_data.csv', header=[0, 1], index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"Error: Make sure you have created the data files. {e}")
        return
    
    # We must convert the string representations of lists in our CSV files back into actual Python lists.
    news_df['tickers'] = news_df['tickers'].apply(ast.literal_eval)

    # 2. Setup the Simulation
    # Load the AI model once at the start to be efficient.
    finbert_tokenizer, finbert_model = setup_finbert_model()

    # Define the simulation's time period and starting capital.
    start_date = news_df['timestamp'].min().date()
    end_date = news_df['timestamp'].max().date()
    start_cash = 100000.0 # 100 racks because thats pocket change. 
    
    # These are our ledgers for tracking performance.
    portfolio = {'cash': start_cash, 'positions': {}} # Tracks current cash and open stock positions.
    trade_log = [] # A detailed log of every trade made.
    portfolio_value_log = [] # A daily log of our total portfolio value.
    
    cash_at_risk_per_trade = 0.05 # We'll risk 5% of the current cash on any new trade. 

    print(f"\nBacktest period: {start_date} to {end_date}")
    print(f"Starting cash: ${start_cash:,.2f}")

    # 3. The Main Simulation Loop 
    # This loop iterates through every single day in our historical period.
    for current_date in pd.date_range(start=start_date, end=end_date):
        date_str = current_date.strftime('%Y-%m-%d')
        
        # a) Manage Open Positions (Check for Exits)
        # Before doing anything else, we check if our open trades have hit their targets.
        for ticker in list(portfolio['positions'].keys()):
            position = portfolio['positions'][ticker]
            try:
                todays_high = price_df.loc[date_str, ('High', ticker)]
                todays_low = price_df.loc[date_str, ('Low', ticker)]

                # Check buy positions
                if position['signal'] == 'BUY':
                    if todays_high >= position['take_profit']:
                        exit_price = position['take_profit']
                        portfolio['cash'] += position['quantity'] * exit_price
                        print(f"  - EXIT {ticker} (TAKE PROFIT) at ${exit_price:.2f}")
                        trade_log.append((current_date, 'EXIT_TP', ticker, position['quantity'], exit_price))
                        del portfolio['positions'][ticker]
                    elif todays_low <= position['stop_loss']:
                        exit_price = position['stop_loss']
                        portfolio['cash'] += position['quantity'] * exit_price
                        print(f"  - EXIT {ticker} (STOP LOSS) at ${exit_price:.2f}")
                        trade_log.append((current_date, 'EXIT_SL', ticker, position['quantity'], exit_price))
                        del portfolio['positions'][ticker]

                # Check sell positions (Shorts)
                elif position['signal'] == 'SELL':
                    if todays_low <= position['take_profit']: # Note: take_profit is a lower price for sells
                        exit_price = position['take_profit']
                        portfolio['cash'] += position['quantity'] * exit_price
                        print(f"  - COVER {ticker} (TAKE PROFIT) at ${exit_price:.2f}")
                        trade_log.append((current_date, 'EXIT_TP', ticker, position['quantity'], exit_price))
                        del portfolio['positions'][ticker]
                    elif todays_high >= position['stop_loss']: # Note: stop_loss is a higher price for sells
                        exit_price = position['stop_loss']
                        portfolio['cash'] += position['quantity'] * exit_price
                        print(f"  - COVER {ticker} (STOP LOSS) at ${exit_price:.2f}")
                        trade_log.append((current_date, 'EXIT_SL', ticker, position['quantity'], exit_price))
                        del portfolio['positions'][ticker]
            except KeyError:
                # This handles weekends/holidays where we might have a position but no price data for that day.
                pass

        # b) Check for new trade signals
        todays_news = news_df[news_df['timestamp'].dt.date == current_date.date()]
        if not todays_news.empty:
            print(f"\nSimulating day: {date_str} - {len(todays_news)} news item(s) found.")
            analysed_news = analyse_sentiment_of_headlines(todays_news, finbert_tokenizer, finbert_model)
            signals = find_trade_signals(analysed_news, confidence_threshold=0.90)
            
            if not signals.empty:
                ranked_signals = rank_signals(signals)
                top_signal = ranked_signals.iloc[0]
                ticker = top_signal['tickers'][0]
                sentiment_signal = top_signal['signal']

                if ticker not in portfolio['positions']:
                    print(f"  - Sentiment signal found: {sentiment_signal} {ticker}")
                    try:
                        # Get the technical signal for confirmation
                        ticker_prices = price_df.xs(ticker, level=1, axis=1).copy()
                        ticker_prices.columns = ticker_prices.columns.str.lower()
                        technical_signal = check_ma_crossover_signal(ticker_prices)
                        print(f"  - Technical signal is: {technical_signal}")
                        
                        if sentiment_signal == technical_signal:
                            print(f"  - CONFIRMED: Signals match. Proceeding with trade.")
                            entry_price = price_df.loc[date_str, ('Open', ticker)]
                            ticker_prices = calculate_atr(ticker_prices)
                            atr = ticker_prices.loc[date_str, 'atr']
                            
                            position_size_cash = portfolio['cash'] * cash_at_risk_per_trade
                            quantity = round(position_size_cash / entry_price)

                            if quantity > 0:
                                cost = quantity * entry_price
                                portfolio['cash'] -= cost
                                portfolio['positions'][ticker] = {
                                    'quantity': quantity, 'entry_price': entry_price, 'entry_date': current_date, 'signal': sentiment_signal,
                                    'stop_loss': entry_price - (atr * 2) if sentiment_signal == 'BUY' else entry_price + (atr * 2),
                                    'take_profit': entry_price + (atr * 4) if sentiment_signal == 'BUY' else entry_price - (atr * 4)
                                }
                                print(f"  - EXECUTED {sentiment_signal}: {quantity} shares of {ticker} at ${entry_price:.2f}")
                                trade_log.append((current_date, 'ENTER', ticker, quantity, entry_price))
                        else:
                            print(f"  - SKIPPING TRADE: Sentiment ({sentiment_signal}) and Technical ({technical_signal}) signals do not match.")
                    except (KeyError, IndexError):
                        print(f"  - Could not find sufficient price data for {ticker} on {date_str}. Skipping trade.")
        
        # c) Record Daily Portfolio Value
        positions_value = 0
        for ticker, pos in portfolio['positions'].items():
            try:
                current_price = price_df.loc[date_str, ('Close', ticker)]
                positions_value += pos['quantity'] * current_price
            except KeyError:
                positions_value += pos['quantity'] * pos['entry_price']
        
        total_value = portfolio['cash'] + positions_value
        portfolio_value_log.append((current_date, total_value))

    # 4. This section is to Analyse and Visualise Results
    print("\n--- Backtest Finished ---")
    final_value = portfolio_value_log[-1][1]
    total_return = (final_value - start_cash) / start_cash * 100
    print("\n--- Backtest Performance ---")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades Made: {len([t for t in trade_log if t[1] == 'ENTER'])}")

    # Create a DataFrame from our daily value log for easy plotting.
    portfolio_df = pd.DataFrame(portfolio_value_log, columns=['Date', 'Value']).set_index('Date')
    
    # Uses matplotlib to create the equity curve chart.
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_df.index, portfolio_df['Value'], label='Sentiment + Technical Strategy')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    run_backtest()