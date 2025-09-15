import feedparser # specialised library designed to make reading and parsing RSS feeds simple.
import pandas as pd

def fetch_rss_feeds(feed_urls: list) -> pd.DataFrame:
    """
    Fetches news articles from a list of RSS feed URLs.

    This function takes a list of web addresses for RSS feeds, reads each one,
    and extracts the key information for each article, like its title and link.
    It then gathers all the articles into a single, clean table (DataFrame).
    """
    # It starts with an empty list that will hold all the articles we find.
    all_news_items = []
    print("Fetching news from RSS feeds...")

    # We then loop through each URL that was provided in the feed_urls list.
    for url in feed_urls:
        # A try-except block is really important for network requests. If a website is down
        # or a link is broken, we don't want our whole bot to crash. This block
        # allows the bot to log the error and simply move on to the next URL.
        try:
            # The feedparser.parse() function downloads the content
            #  from the URL and interprets the RSS feed's structure.
            feed = feedparser.parse(url)
            
            # The 'feed' object now contains all the information. The 'entries'
            # attribute is a list, where each item is a single news article.
            for entry in feed.entries:
                # We create a Python dictionary for each article to store the
                # information we care about in a structured way.
                news_item = {
                    # 'feed.feed.title' gives us the name of the news source itself.
                    'source': feed.feed.get('title', 'N/A'),
                    # 'entry.title' is the headline of the specific article.
                    'title': entry.get('title', 'N/A'),
                    # 'entry.link' is the direct web address for the full article.
                    'link': entry.get('link', 'N/A'),
                    'published': pd.to_datetime(entry.get('published', None), utc=True)
                    # We convert the publication date into a standardised pandas datetime
                    # format and set it to the UTC timezone. This prevents timezone-related
                    # bugs, which I encountered during development :( 
                }
                # We add our newly created dictionary to our master list.
                all_news_items.append(news_item)
            
            print(f"  - Successfully fetched {len(feed.entries)} items from {url}")

        except Exception as e:
            # If anything went wrong in the 'try' block, this code runs.
            print(f"  - Failed to fetch or parse {url}. Error: {e}")

    if not all_news_items:
        print("No news items were fetched from RSS feeds.")
        return pd.DataFrame()

    # Code below is for some Data Cleaning and structuring. 
    # Convert our list of dictionaries into a clean pandas DataFrame.
    df = pd.DataFrame(all_news_items)

    # Remove any rows that might have been missing a publication date.
    df.dropna(subset=['published'], inplace=True)
    # Sort the entire table by date, so the most recent news is always at the top.
    df.sort_values(by='published', ascending=False, inplace=True)
    # Reset the index to be a clean 0, 1, 2, ... sequence after sorting.
    df.reset_index(drop=True, inplace=True)

    return df