import torch #The main library for PyTorch, which is the deep learning framework the FinBERT model is built on.
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
# key library from Hugging Face that gives us access to pre-trained models like FinBERT.


def setup_finbert_model():
    """
    Loads the FinBERT model and its tokenizer.

    This function is designed to be robust. It first tries to load our custom,
    fine-tuned model from the local directory, which was optimised. If that folder doesn't exist
    (or there's an error), it has a fallback. It will download and load the
    original, generic FinBERT model from the internet. This ensures the bot can
    always run.
    """
    model_path = './my_custom_finbert_model'
    print(f"Attempting to load CUSTOM FinBERT model from {model_path}")
    
    try:
        # Load the tokenizer and model from your local, fine-tuned folder.
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Custom model loaded successfully")
    except OSError:
        # This code runs if the 'my_custom_finbert_model' folder is not found.
        print(f"Custom model not found at {model_path}. Falling back to default model.")
        default_path = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(default_path)
        model = AutoModelForSequenceClassification.from_pretrained(default_path)

    return tokenizer, model

def analyse_sentiment_of_headlines(news_df: pd.DataFrame, tokenizer, model) -> pd.DataFrame:
    """
    Analyses the sentiment of each headline in the provided DataFrame.
    This function uses "batch processing" to analyse many headlines without
    running out of memory, which is essential for running on a local machine.
    """
    if news_df.empty:
        return news_df

    headlines = news_df['title'].tolist()
    batch_size = 16 # We'll process 16 headlines at a time. 16 was picked because of issues with
    # RAM when running locally. This can be increased depending on hardware. 

    # We'll store the results for all batches in these lists.
    all_sentiments = []
    all_confidences = []
    
    # This loop iterates through our list of headlines, one batch at a time.
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i+batch_size]
        
        # The tokenizer converts the batch of text headlines
        # into a format of numbered tokens that the AI model can understand.
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        # 'torch.no_grad()' is a performance optimisation. It tells PyTorch that
        # we are only doing inference (making predictions), not training, so it
        # doesn't need to calculate gradients, which saves memory and computation.
        with torch.no_grad():
            # The tokenized text is fed into the model.
            # The model outputs its raw, uncalibrated scores, called 'logits'.
            outputs = model(**inputs)

        # Softmax: The softmax function converts the raw logits into a set of
        # probabilities for each class (positive, negative, neutral) that sum to 1.
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Now we find the label with the highest probability for each headline.
        labels = ["positive", "negative", "neutral"]
        # `torch.argmax` finds the index of the highest probability (0, 1, or 2).
        batch_sentiments = [labels[torch.argmax(p)] for p in predictions]
        # `torch.max` finds the value of the highest probability (our confidence score).
        batch_confidences = [torch.max(p).item() for p in predictions]
        
        # Add the results of this batch to our master lists.
        all_sentiments.extend(batch_sentiments)
        all_confidences.extend(batch_confidences)

    # Add the complete lists of results as new columns to the DataFrame.
    analysed_df = news_df.copy()
    analysed_df['sentiment'] = all_sentiments
    analysed_df['confidence'] = all_confidences

    return analysed_df