import time
import pandas as pd
from alpaca.trading.client import TradingClient

# Imports all the functions from our modules in their respective folders. 
from bot_modules.news_manager import gather_all_news
from bot_modules.identifier import define_stock_universe, tag_headlines_with_tickers
from bot_modules.analyser import setup_finbert_model, analyse_sentiment_of_headlines
from bot_modules.decision_engine import find_trade_signals, rank_signals
from bot_modules.executor import get_broker_api, execute_trade_signal

def run_live_pipeline(finbert_tokenizer, finbert_model, stock_universe, trading_api, seen_news_links):
    """
    This function represents a single check-in cycle of the live trading bot.
    It executes the full pipeline from getting the current state to potentially executing a trade.
    """
    print(f"\n--- Running pipeline at {time.ctime()}")
    
    # (1) First thing is to get current portfolio state
    # So that the bot knows the current situation.
    try:
        positions = trading_api.get_all_positions()
        # We create a dictionary for fast lookup of our current stock holdings.
        current_positions = {p.symbol: p for p in positions}
        account_details = trading_api.get_account()
        portfolio_value = float(account_details.portfolio_value)
        print(f"Current positions: {list(current_positions.keys()) or 'None'}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
    except Exception as e:
        print(f"Error fetching portfolio state: {e}")
        return seen_news_links # Stop this cycle if we can't get portfolio state.

    # (2) This section now gathers and filters news
    all_news_df = gather_all_news()
    if all_news_df.empty:
        print("No news was gathered in this cycle.")
        return seen_news_links

    # This is a key part of state management. We filter out any articles whose 'link'
    # is already in our `seen_news_links` memory set. The '~' is a logical NOT.
    new_articles_df = all_news_df[~all_news_df['link'].isin(seen_news_links)]
    if new_articles_df.empty:
        print("No new news articles found in this cycle.")
        return seen_news_links
    
    print(f"Found {len(new_articles_df)} new articles to analyse.")
    
    # (3) This section here runs the "brain" of our bot (analyser and decision_engine)
    tagged_df = tag_headlines_with_tickers(new_articles_df.copy(), stock_universe)
    relevant_news_df = tagged_df[tagged_df['tickers'].apply(lambda x: len(x) > 0)].copy()

    if not relevant_news_df.empty:
        analysed_df = analyse_sentiment_of_headlines(relevant_news_df, finbert_tokenizer, finbert_model)
        trade_signals_df = find_trade_signals(analysed_df, confidence_threshold=0.85)
        
        if not trade_signals_df.empty:
            ranked_signals_df = rank_signals(trade_signals_df)
            top_signal = ranked_signals_df.iloc[0]
            ticker_to_trade = top_signal['tickers'][0]

            # (4) This section is for risk management and trade execution
            # Before executing any trade, we perform several risk checks.
            max_allocation_per_stock = 0.10  # Max 10% of portfolio in any single stock. 
            cash_for_new_trade = portfolio_value * 0.05 # Use 5% of portfolio value for new trades. 
            # The two values above, I just selected and can be adjusted as required. 
            current_position_value = 0.0
            if ticker_to_trade in current_positions:
                print(f"Signal found for {ticker_to_trade}, which is already in the portfolio.")
                current_position_value = float(current_positions[ticker_to_trade].market_value)

            # The final check to see if this new trade breach our risk limit
            if (current_position_value + cash_for_new_trade) <= (portfolio_value * max_allocation_per_stock):
                print("\n--- Top Trade Signal Identified ---")
                print(top_signal[['signal', 'confidence', 'tickers', 'title']])
                
                # (5) FInally executes the trade via the broker API (in this case alpaca)
                execute_trade_signal(trading_api, top_signal, cash_to_use=cash_for_new_trade)
            else:
                print(f"SKIPPING TRADE: Adding ${cash_for_new_trade:,.2f} to {ticker_to_trade} would exceed the {max_allocation_per_stock:.0%} allocation limit.")

    # (6) This section updates the bot's memory with the news it has just processed
    # We add all the links from this cycle's news batch to our memory set
    # so they will be ignored in the future.
    for link in new_articles_df['link']:
        seen_news_links.add(link)
        
    return seen_news_links

def main():
    """
    The main function that initialises and runs the live bot in a continuous loop.
    """
    # We set up all the components (AI model, API connections) once at the start
    # to be efficient. We don't want to reload the huge AI model every 10 minutes.
    print("--- Initializing Live Trading Bot ---")
    finbert_tokenizer, finbert_model = setup_finbert_model()
    stock_universe = define_stock_universe()
    trading_api = get_broker_api()
    
    # This set will act as the bot's long-term memory for news articles.
    seen_news_links = set()
    
    # This is the big cycle that keeps the bot running indefinitely until stopped with CTRL+C in the terminal. 
    # This `while True:` loop will run the entire pipeline, then sleep for a set time, then repeat.
    while True:
        try:
            # We pass the bot's memory (`seen_news_links`) into the pipeline and get the
            # updated version back to use in the next cycle.
            seen_news_links = run_live_pipeline(finbert_tokenizer, finbert_model, stock_universe, trading_api, seen_news_links)
            
            # This is the schedule. The bot will pause (sleep) for a set number of minutes.
            sleep_duration_minutes = 10
            print(f"\n--- Cycle complete. Sleeping for {sleep_duration_minutes} minutes... ---")
            time.sleep(sleep_duration_minutes * 60)
            
        except KeyboardInterrupt:
            # This allows us to stop the bot cleanly by pressing Ctrl+C in the terminal.
            print("\nBot stopped manually.")
            break # Exit the `while True` loop.
        except Exception as e:
            # This is a critical safety net. If any unexpected error occurs, the bot
            # won't crash. It will print the error, wait 5 minutes, and then try again.
            print(f"\nAn error occurred in the main loop: {e}")
            print("Restarting in 5 minutes...")
            time.sleep(300)

# This ensures the `main()` function is only called when we run `python live_bot.py` directly.
if __name__ == '__main__':
    main()