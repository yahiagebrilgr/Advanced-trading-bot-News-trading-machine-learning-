import pandas as pd
from bot_modules.news_manager import gather_all_news

def collect_headlines_for_labeling():
    """
    Runs the live news gathering pipeline for the purpose of collecting
    a large number of unique, real-world headlines. The output CSV from
    this script is the starting point for the manual labeling process
    required to create a custom training dataset.
    """
    print("--- Gathering headlines for labeling ---")
    all_news_df = gather_all_news()
    if all_news_df.empty:
        print("No news was gathered.")
        return
    unique_titles = all_news_df.drop_duplicates(subset=['title'])
    file_path = 'headlines_to_label.csv' # This file is temporary
    unique_titles[['title']].to_csv(file_path, index=False)
    print(f"\nSuccessfully saved {len(unique_titles)} unique headlines to {file_path}")

if __name__ == '__main__':
    collect_headlines_for_labeling()