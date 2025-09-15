import pandas as pd

def generate_historical_news_dataset():
    """
    Creates a sample CSV file of historical news for US stocks.
    This file acts as the "News Timeline" for the backtester, providing the
    events that will trigger the bot's decisions.
    """
    print("--- Creating US historical news dataset ---")
    # This list can be expanded with more real or simulated news events.
    news_data = [
        # January 2023
        {'timestamp': '2023-01-05 09:01:00', 'title': "Apple suppliers gear up for new iPhone production", 'tickers': "['AAPL']", 'sentiment': 'positive', 'confidence': 0.96},
        {'timestamp': '2023-01-05 14:30:00', 'title': "Analysts raise concerns over Tesla's delivery numbers", 'tickers': "['TSLA']", 'sentiment': 'negative', 'confidence': 0.91},
        {'timestamp': '2023-01-06 11:00:00', 'title': "Microsoft faces new antitrust probe in Europe", 'tickers': "['MSFT']", 'sentiment': 'negative', 'confidence': 0.98},
        {'timestamp': '2023-01-09 10:15:00', 'title': "Google parent Alphabet announces major AI breakthrough", 'tickers': "['GOOGL']", 'sentiment': 'positive', 'confidence': 0.97},
        {'timestamp': '2023-01-10 09:30:00', 'title': "Nvidia stock soars on new GPU announcement", 'tickers': "['NVDA']", 'sentiment': 'positive', 'confidence': 0.99},
        {'timestamp': '2023-01-18 12:00:00', 'title': "Amazon to hire 50,000 new workers for AWS expansion", 'tickers': "['AMZN']", 'sentiment': 'positive', 'confidence': 0.93},
        {'timestamp': '2023-01-25 16:00:00', 'title': "Tesla cuts prices globally in push for volume", 'tickers': "['TSLA']", 'sentiment': 'negative', 'confidence': 0.95},

        # February 2023
        {'timestamp': '2023-02-02 08:45:00', 'title': "Strong iPhone sales push Apple earnings past estimates", 'tickers': "['AAPL']", 'sentiment': 'positive', 'confidence': 0.98},
        {'timestamp': '2023-02-03 10:00:00', 'title': "Amazon reports disappointing quarter, stock tumbles", 'tickers': "['AMZN']", 'sentiment': 'negative', 'confidence': 0.97},
        {'timestamp': '2023-02-08 11:30:00', 'title': "Microsoft's new AI-powered Bing challenges Google", 'tickers': "['MSFT', 'GOOGL']", 'sentiment': 'neutral', 'confidence': 0.85},
        {'timestamp': '2023-02-15 14:00:00', 'title': "Nvidia earnings beat expectations on data center growth", 'tickers': "['NVDA']", 'sentiment': 'positive', 'confidence': 0.96},
        {'timestamp': '2023-02-22 09:00:00', 'title': "Federal regulators block major Microsoft acquisition", 'tickers': "['MSFT']", 'sentiment': 'negative', 'confidence': 0.99},

        # March 2023
        {'timestamp': '2023-03-01 10:10:00', 'title': "Tesla investor day leaves shareholders wanting more", 'tickers': "['TSLA']", 'sentiment': 'negative', 'confidence': 0.92},
        {'timestamp': '2023-03-07 13:00:00', 'title': "Google launches new cloud services for enterprise", 'tickers': "['GOOGL']", 'sentiment': 'positive', 'confidence': 0.94},
        {'timestamp': '2023-03-15 09:45:00', 'title': "Apple rumored to be working on a new VR headset", 'tickers': "['AAPL']", 'sentiment': 'positive', 'confidence': 0.91},
        {'timestamp': '2023-03-23 16:00:00', 'title': "US government sues Amazon over monopoly claims", 'tickers': "['AMZN']", 'sentiment': 'negative', 'confidence': 0.98}
    ]
    df = pd.DataFrame(news_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    file_path = 'data/historical_news_dataset.csv' # Corrected path
    df.to_csv(file_path, index=False)
    print(f"Successfully created US dataset with {len(df)} entries.")
    print(f"File saved to: {file_path}")

if __name__ == '__main__':
    generate_historical_news_dataset()






    