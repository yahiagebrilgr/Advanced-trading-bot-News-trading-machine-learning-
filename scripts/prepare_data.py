import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    """
    Loads the full manually labeled dataset (`labeled_news_dataset.csv`)
    and splits it into a training set (80%) and a testing set (20%).
    This is a critical step to prevent model overfitting, ensuring we can
    test the final model on data it has never seen before.
    """
    print("--- Preparing data for fine-tuning ---")
    try:
        # Load the large dataset you created
        df = pd.read_csv('labeled_news_dataset.csv')
    except FileNotFoundError:
        print("Error: `labeled_news_dataset.csv` not found. Please create it first.")
        return

    # A 80/20 split is a standard practice in machine learning
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,      # 20% of the data will be for testing
        random_state=42,    # Ensures the split is the same every time we run it
        stratify=df['label'] # Ensures the proportion of pos/neg/neu is the same in both sets
    )

    # Save the new datasets to separate files
    train_df.to_csv('data/train_dataset.csv', index=False)
    test_df.to_csv('data/test_dataset.csv', index=False)

    print(f"Data split complete.")
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    print("Files `train_dataset.csv` and `test_dataset.csv` created.")

if __name__ == '__main__':
    split_dataset()