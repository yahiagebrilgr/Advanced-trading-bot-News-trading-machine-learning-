import os #This is to interact with environment variables.
import requests # imported for HTTP requests
import pandas as pd # For data manipulation and analysis, used for structured table
from dotenv import load_dotenv #for .env file handling for my API Keys for alpaca and marketaux. 

def fetch_api_news(symbols: list, api_key: str) -> pd.DataFrame:
    """
    Fetches news for a specific list of stock symbols from the MarketAux API.
    """
    # Firstly a safety check. If the API key wasn't loaded correctly,
    # An error is printedand returns an empty DataFrame to prevent the script from crashing.
    if not api_key:
        print("Error: MARKETAUX_API_KEY not found. Please check your .env file.")
        return pd.DataFrame()

    
    # Code below is for requests. This is the "conversation" we have with the API.
    # This is the permanent web address of the API service we want to talk to.
    base_url = "https://api.marketaux.com/v1/news/all"
    
    # This is a Python dictionary where we build our specific "order" for the API,
    # telling it exactly what data we want back.
    params = {
        # This tells the API which stock symbols we are interested in.
        'symbols': ','.join(symbols),
        'filter_entities': 'true',
        'language': 'en',
        # We provide our secret API key to prove we have permission to ask for data.
        'api_token': api_key
    }

    print(f"Fetching news from API for symbols: {', '.join(symbols)}...")
    all_news_items = []

    # Sending the Request and Getting the Response from the API 

    # A try-except block is very important for any code that accesses the internet.
    # It allows us to handle network errors without crashing the whole bot.
    try:
        # This is the line that actually sends our request over the internet.
        # The requests library combines the base_url and our params into a full URL.
        response = requests.get(base_url, params=params)
        
        # This is a built-in safety check. If the website returns an error (like 404 Not Found),
        # this line will automatically raise an error that our except block can catch.
        response.raise_for_status()

        # The .json() method automatically converts this text-based data into a Python dictionary.
        # The .get('data', []) is a good way to access the list of articles. If the 'data' key
        # doesn't exist for some reason, it will give us an empty list [] instead of crashing.
        data = response.json().get('data', [])

        # We loop through each article the API sent back.
        for item in data:
            # We create a dictionary for each article, carefully structuring it to match
            # the output from our RSS scanner. 
            news_item = {
                'source': item.get('source', 'N/A'),
                'title': item.get('title', 'N/A'),
                'link': item.get('url', 'N/A'),
                'published': pd.to_datetime(item.get('published_at', None), utc=True)
            }
            all_news_items.append(news_item)
        
        print(f"  - Successfully fetched {len(data)} items from the API.")

    except requests.exceptions.RequestException as e:
        # If any network error occurred, this code will run.
        print(f"  - Failed to fetch from API. Error: {e}")
        return pd.DataFrame()

    if not all_news_items:
        return pd.DataFrame()

    
    # Code below is for data cleaning and structuring.
    # Convert our list of dictionaries into a clean, structured pandas DataFrame.
    df = pd.DataFrame(all_news_items)
    # Remove any rows that might be missing a publication date.
    df.dropna(subset=['published'], inplace=True)
    # Sort the articles with the newest ones at the top.
    df.sort_values(by='published', ascending=False, inplace=True)
    # Reset the DataFrame index to be a clean 0, 1, 2, ... sequence.
    df.reset_index(drop=True, inplace=True)
    
    return df