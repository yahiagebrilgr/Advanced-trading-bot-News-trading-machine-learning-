import pandas as pd
from sklearn.metrics import accuracy_score, classification_report 
# We import tools from scikit-learn, a major machine learning library, to measure performance.
from bot_modules.analyser import analyse_sentiment_of_headlines

def setup_model_from_path(model_path="ProsusAI/finbert"):
    """ A helper function to load a model from a specific path. """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f" Loading model from: {model_path} ")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("Model loaded successfully ")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model from {model_path}. Error: {e}")
        return None, None

def evaluate():
    """
    Compares the performance of the original FinBERT against our custom model
    on the unseen test dataset to see which one is better.
    """
    print(" Evaluating Model Performance ")

    # Load the final exam data that neither model has seen before. (20% was used in my case)
    test_df = pd.read_csv('data/test_dataset.csv')
    true_labels = test_df['label'].tolist()
    
    # 1. Grade the OG FinBERT model 
    original_tokenizer, original_model = setup_model_from_path("ProsusAI/finbert")
    if original_model:
        original_results = analyse_sentiment_of_headlines(test_df.copy(), original_tokenizer, original_model)
        original_predictions = original_results['sentiment'].tolist()
        
        # accuracy_score gives us a single percentage of correct answers.
        original_accuracy = accuracy_score(true_labels, original_predictions)
        
        print("\n--- Original FinBERT Performance ---")
        print(f"Accuracy: {original_accuracy:.2%}")
        # classification_report gives a more detailed breakdown, showing how well
        # the model performed on each specific category (positive, negative, neutral).
        print(classification_report(true_labels, original_predictions))

    # 2. Grade the custom/trained FinBERT model 
    custom_tokenizer, custom_model = setup_model_from_path('./my_custom_finbert_model')
    if custom_model:
        custom_results = analyse_sentiment_of_headlines(test_df.copy(), custom_tokenizer, custom_model)
        custom_predictions = custom_results['sentiment'].tolist()
        custom_accuracy = accuracy_score(true_labels, custom_predictions)
        
        print("\n The Custom FinBERT Performance ")
        print(f"Accuracy: {custom_accuracy:.2%}")
        print(classification_report(true_labels, custom_predictions))

if __name__ == '__main__':
    evaluate()


# If you run this script, you will see that the new custom 
# machine learning model is much more accurate than the original 
# Original finBERT was only 68.92% accurate, whereas the new and improved finBERT is 85.56% accurate.
# EVO-FinBERT  