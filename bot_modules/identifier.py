import pandas as pd

def define_stock_universe() -> dict:
    """
    Defines the master list of US stocks that our bot is interested in.

    The "universe" is the pre-selected group of securities
    that a strategy is allowed to trade from. 
    
    Returns:
        A dictionary where:
        - The KEY is the official stock ticker (for ex. 'AAPL') use for trading and getting price data.
        - The VALUE is a list of lowercase search terms to find that company in news headlines.

    The list of stocks here is a small sample, and can be expanded/modified. US/Global stocks were selected
    as they are more widely covered by the news APIs and RSS feeds that are used in this project. Furthermore,
    it helps when trading in alpaca as its a US based broker. 
    """
    universe = {
        # Tech
        'AAPL': ['apple', 'iphone'],
        'MSFT': ['microsoft', 'windows'],
        'GOOGL': ['google', 'alphabet'],
        'TSLA': ['tesla', 'elon musk'],
        'NVDA': ['nvidia', 'gpu', 'gpus'],
        'AMZN': ['amazon', 'aws'],
        'META': ['meta', 'facebook', 'instagram'],
        'NFLX': ['netflix'],
        'INTC': ['intel'],
        'SHOP': ['shopify'],

        # Finance
        'JPM': ['jpmorgan', 'jpmorgan chase'],
        'BAC': ['bank of america'],
        'V': ['visa'],
        'MA': ['mastercard'],

        # Consumer & Retail
        'WMT': ['walmart'],
        'COST': ['costco'],
        'NKE': ['nike'],
        'MCD': ["mcdonald's"],
        'KO': ['coca-cola', 'coke'],
        'DIS': ['disney', 'walt disney'],
        
        # Healthcare
        'JNJ': ['johnson & johnson', 'j&j'],
        'PFE': ['pfizer'],
        'MRNA': ['moderna'],

        # Industrial, Energy & Defense
        'CVX': ['chevron'],
        'XOM': ['exxonmobil', 'exxon'],
        'BA': ['boeing'],
        'LMT': ['lockheed martin', 'lockheed'],
        'NOC': ['northrop grumman', 'northrop'],
        'RTX': ['rtx', 'raytheon'],
        'GD': ['general dynamics'],

        # Telecom
        'T': ['at&t']
    }
    return universe

def tag_headlines_with_tickers(news_df: pd.DataFrame, stock_universe: dict) -> pd.DataFrame:
    """
    Scans news headlines and tags them with stock tickers from our universe.
    This is a simple but effective form of Named Entity Recognition (NER).
    """
    # We make a copy of the input DataFrame to avoid modifying the original data. 
    # without the .copy(), there was errors during development. 
    news_df = news_df.copy()
    
    # This list will store our results, one item for each headline.
    tagged_tickers_list = []
    
    # We loop through each individual headline in the title column of the DataFrame.
    for title in news_df['title']:
        found_tickers_for_this_headline = []
        
        # This is an important step for robust matching. We convert the headline to lowercase.
        # This ensures that "Apple", "apple", and "APPLE" are all treated the same.
        # We also convert it to a string with str() to prevent errors if a title is not text.
        title_lower = str(title).lower()
        
        # For each headline, we check against every single stock in our universe.
        for ticker, search_terms in stock_universe.items():
            # And for each stock, we check all of its possible search terms.
            for term in search_terms:
                # The 'in' keyword performs a simple text search.
                if term in title_lower:
                    # If we find a match, we add the official ticker to our list.
                    found_tickers_for_this_headline.append(ticker)
                    # This 'break' is a small optimisation. Once we've found 'apple',
                    # we don't need to keep checking for 'iphone' for the same headline,
                    # so we break out of this inner loop and move to the next stock.
                    break 
        
        # We add the list of tickers found for this one headline to our master list.
        # The list might be empty if no matches were found.
        tagged_tickers_list.append(found_tickers_for_this_headline)

    # We create a brand new 'tickers' column in our DataFrame.
    # Each entry in this column is a list, because a single headline might
    # mention multiple companies (for ex. "Microsoft and Google announce partnership").
    news_df['tickers'] = tagged_tickers_list
    return news_df