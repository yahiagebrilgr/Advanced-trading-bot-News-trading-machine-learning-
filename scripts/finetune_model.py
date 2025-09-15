import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def run_finetuning():
    """
    Loads the pre-trained FinBERT model and fine-tunes it on our custom labeled dataset.
    This process creates a new, specialized model saved locally.
    """
    print("--- Starting FinBERT Fine-Tuning Process ---")

    # 1. Load the custom datasets created by prepare_data.py
    train_df = pd.read_csv('data/train_dataset.csv')
    test_df = pd.read_csv('data/test_dataset.csv')

    # The model works with numbers, not text labels. We create a map to convert them.
    # positive -> 0, negative -> 1, neutral -> 2
    label_map = {'positive': 0, 'negative': 1, 'neutral': 2}
    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    # Convert our pandas DataFrames into a special format that the Hugging Face library prefers.
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 2. Load the base model and tokenizer we want to improve upon.
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # 3. Tokenize the datasets
    # This function takes our text and converts it into numbers the model can understand.
    def tokenize_function(examples):
        return tokenizer(examples['title'], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # 4. Define Training Arguments
    # This is like setting up the "syllabus" for the training process.
    training_args = TrainingArguments(
        output_dir='./results',          # Where to save checkpoints during training.
        num_train_epochs=3,              # How many times to go through the entire training manual.
        per_device_train_batch_size=2,   # How many examples to show the model at once (kept low for memory).
        gradient_accumulation_steps=4,   # A technique to simulate a larger batch size without using more memory.
        warmup_steps=500,                # How many initial steps to gradually increase the learning rate.
        weight_decay=0.01,               # A regularization technique to prevent overfitting.
        logging_dir='./logs',            # Where to save training logs.
        eval_strategy="epoch"            # How often to run the "final exam", right after each epoch in this case. 
    )

    # 5. Create the Trainer
    # The Trainer is a  Hugging Face class that handles the entire training loop for us.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )

    print("\nStarting Training")
    trainer.train()
    print("Training Complete")

    # 6. Save the new and improved model to a permanent location.
    save_path = './my_custom_finbert_model'
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path) # Also save the tokenizer with the model.
    print(f"Fine-tuned model saved to {save_path}")

if __name__ == '__main__':
    run_finetuning()