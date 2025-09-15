import os
import pandas as pd
from dotenv import load_dotenv

# The '.' before the module name is for relative imports
from .identifier import define_stock_universe
from .rss_scanner import fetch_rss_feeds
from .api_client import fetch_api_news

def gather_all_news() -> pd.DataFrame:
    """
    This function acts as the orchestrator for all news gathering.
    It calls the modules to fetch news from different sources,
    then combines and cleans the data into a single, unified DataFrame.
    """
    print("Starting News Gathering Process")
    
    # Below fetchs broad news from RSS Feed
    # The idea of this is to start by casting a wide net with general financial news feeds.
    # This helps us capture broader market sentiment and stories we might not
    # specifically search for.
    rss_urls = [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://www.investing.com/rss/news_25.rss"
    ]
    rss_df = fetch_rss_feeds(rss_urls)

    # The code below fetch targeted news from the API
    # We then perform a targeted search for news related only to the stocks
    # in our universe. To make sure we don't miss any critical, company-specific news.
    load_dotenv()
    api_key = os.getenv("MARKETAUX_API_KEY")
    stock_universe = define_stock_universe()
    api_symbols = list(stock_universe.keys())
    api_df = fetch_api_news(symbols=api_symbols, api_key=api_key)

    # The section below combines and cleans the data from both sources (RSS and news API)
    print("\nCombining all news sources...")
    # `pd.concat` is a pandas function that stacks DataFrames on top of each other.
    # `ignore_index=True` is important because it tells pandas to create a fresh
    # index (0, 1, 2, and so on) for the new, combined DataFrame.
    combined_df = pd.concat([rss_df, api_df], ignore_index=True)
    print(f"Total items before de-duplication: {len(combined_df)}")

    # De-duplication is a important for data cleaning in this case. The same breaking news story
    # will appear on multiple feeds. We only want to analyse it once.
    # `subset=['title']` tells pandas to consider a row a duplicate if its 'title' is identical to another's.
    # `keep='first'` tells pandas to keep the first instance it finds and discard the subsequent ones.
    # The `.copy()` prevents a technical warning from pandas about modifying a 'slice' of a DataFrame.
    final_df = combined_df.drop_duplicates(subset=['title'], keep='first').copy()
    
    # This bit below ensures the data is perfectly ordered and indexed.
    final_df.sort_values(by='published', ascending=False, inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    print(f"Total unique items after de-duplication: {len(final_df)}")
    print("News Gathering Process Complete")

    return final_df