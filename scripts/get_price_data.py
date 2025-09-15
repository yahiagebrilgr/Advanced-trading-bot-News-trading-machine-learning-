## This file downloads historical price data for all stocks in our universe, so that it can be used for backtesting
## This goes in hand with the news timeline
import yfinance as yf
import pandas as pd
# We import our own function to get the list of stocks we need to download data for.
from bot_modules.identifier import define_stock_universe

def download_historical_price_data():
    """
    Downloads historical daily price data for all stocks in our universe
    and saves it to a local CSV file for the backtester to use.
    """
    print("--- Downloading historical price data ---")

    stock_universe = define_stock_universe()
    tickers = list(stock_universe.keys())

    # Define the time period for the historical data.
    # This should always cover the dates in our news dataset.
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")

    try:
        # yfinance.download can fetch data for multiple tickers at once.
        # auto_adjust=True is important: it automatically adjusts the prices for
        # stock splits and dividends, giving us a cleaner dataset to work with.
        price_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

        # A check to ensure the download was successful and returned data.
        if price_data.empty:
            print("\nError: No data was downloaded. Tickers might be incorrect or no data available for the period.")
            return

        # We can drop the 'Volume' column to keep our data file smaller, as our
        # current strategy doesn't use it. 'level=0' is needed because the columns have multiple levels.
        try:
            price_data.drop(columns=['Volume'], inplace=True, level=0)
        except KeyError:
            print("Warning: 'Volume' column not found to drop. Proceeding without it.")

        file_path = 'data/historical_price_data.csv'
        price_data.to_csv(file_path)

        print(f"\nSuccessfully downloaded price data.")
        print(f"File saved to: {file_path}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}")

# This makes the script runnable from the command line.
if __name__ == '__main__':
    download_historical_price_data()