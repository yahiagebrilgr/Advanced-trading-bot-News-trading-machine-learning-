# AI-Powered News Sentiment Trading Bot with a Custom-Trained FinBERT Model

## Project Aim

This project is the culmination of a series of explorations into quantitative trading, building directly on the foundations of my last 3 previous projects. The primary goal was to build a fully autonomous, multi-asset trading bot from the ground up, capable of making its own trading decisions by analysing real-time financial news.

The key learning objective was to move beyond theory and implement a practical, end-to-end system that could:
* Integrate and use a sophisticated, pre-trained NLP model (FinBERT) for a real-world task.
* Connect to a live brokerage API (Alpaca) to manage a paper trading account.
* Combine fundamental signals (news sentiment) with technical analysis (a moving average filter) to create a more robust, hybrid trading strategy.
* And finally, to take on some machine learning by creating a custom dataset and fine-tuning the AI model to significantly improve its performance.


## How It Works: The Five-Stage Pipeline

The bot's logic is structured as a five-stage pipeline that turns raw information from the internet into a live, risk-managed trade.

* **Stage 1: News Gathering**
    The bot starts by casting a wide net, scanning multiple general financial news RSS feeds and a targeted news API to gather a large list of the latest headlines. All incoming news is consolidated and de-duplicated to ensure each story is only analysed once.

* **Stage 2: Identification**
    This raw feed is then filtered. The bot scans each headline for keywords related to a pre-defined "universe" of stocks (I manually entered a selection of US and International stocks). Any article that doesn't mention a stock from our universe is discarded, focusing the bot's attention only on relevant news.

* **Stage 3: AI Analysis**
    The relevant headlines are passed to the bot's "brain", which is the custom-trained FinBERT model. The model reads each headline and makes a judgment, assigning it a sentiment (`positive`, `negative`, or `neutral`) and a confidence score in its own prediction.

* **Stage 4: Decision Engine & Portfolio Management**
    The bot's command centre takes the AI's analysis and applies a strict set of rules. It first filters for high-conviction signals (for ex. sentiment that is not neutral and has a confidence score above 85%). It then applies a final technical filter, checking the stock's market trend using a moving average crossover. A trade is only confirmed if the positive or negative news sentiment aligns with the current price trend. The bot also checks its live portfolio to manage risk, ensuring that no single stock exceeds a 10% allocation of the total portfolio value.

* **Stage 5: Execution & Risk Management**
    If a trade signal is confirmed, the bot connects to the Alpaca paper trading account. Before placing an order, it calculates the Average True Range (ATR), which is a measure of the stock's recent volatility. It uses this ATR to automatically set dynamic stop-loss and take-profit targets for the trade. The final order is submitted as a bracket order, which is a three-part order that includes the entry, the take-profit target, and the stop-loss, which ensures every trade is automatically managed from the moment it's placed.

---

## Optimising the AI model: A Custom-Trained FinBERT Model

An important part of this project was to move beyond just using off-the-shelf tools and to perform a full machine learning optimisation cycle. The standard FinBERT model is a powerful generalist, but I wanted to create a specialist.

The process involved:
1.  **Data Creation:** I created a large, custom dataset of over 800 financial news headlines. This was a long manual process where I read each headline and assigned my own `positive`, `negative`, or `neutral` label to create a high-quality dataset.
2.  **Model Training:** To prevent overfitting, the dataset was split into an 80% training set and a 20% testing set. Using Google Colab for its free GPU resources, I took the original FinBERT model and fine-tuned it on my custom training data. This process re-calibrated the model's parameters, teaching it the specific nuances and vocabulary of the news that my strategy focuses on.
3.  **Evaluation:** Before deploying the new model, it was tested on the "unseen" testing set.

The results showed a dramatic improvement in performance. The **accuracy of the model jumped from 68.92% to 87.84%**. The most significant improvement was in its ability to correctly identify neutral, irrelevant news, which is crucial for helping the bot filter out market noise and avoid bad trades. This custom-trained model is what now powers the bot.

---

## Key Features

* **Multi-Source News Aggregation:** Gathers and consolidates news from both general RSS feeds and a targeted API.
* **Custom-Trained AI Brain:** Powered by a fine-tuned FinBERT model that was optimised on a custom dataset for a ~19% improvement in accuracy.
* **Hybrid Trading Strategy:** Fuses fundamental news sentiment with a technical moving average trend filter for more robust, confirmed trading signals.
* **Event-Driven Backtester:** Includes a custom-built simulator to test the strategy's historical performance.
* **Live Paper Trading:** The final application is a fully autonomous bot that can run continuously, manage its state, and trade live in a paper account.
* **Modular Codebase:** The project is organized into a clean, professional structure, separating the core logic from the runnable applications.

---

## Project Structure

The project is organised into a clean, modular structure to separate concerns and make the code easy to maintain and understand.

/AI_trade_bot_ALL_Stocks_V1/
├── bot_modules/            # Contains all the core, reusable logic for the bot.
│   ├── api_client.py       # Fetches targeted news from the MarketAux API.
│   ├── analyser.py         # The "brain": loads and runs the AI model for sentiment analysis.
│   ├── decision_engine.py  # Contains the strategy's rules for filtering and ranking signals.
│   ├── executor.py         # The "hands": connects to the broker and executes trades.
│   ├── identifier.py       # Defines the stock universe and finds relevant news.
│   ├── news_manager.py     # Completes the gathering and cleaning of all news.
│   └── rss_scanner.py      # Fetches general news from RSS feeds.
│
├── data/                   # Stores all the data files used for backtesting and training.
│
├── scripts/                # Contains one-off utility scripts for setup and analysis.
│   ├── create_dataset.py   # Generates the historical news file for the backtester.
│   ├── evaluate_model.py   # Compares the performance of the original vs. custom AI model.
│   ├── finetune_model.py   # The script used to train our custom AI model.
│   └── ... (and other data preparation scripts)
│
├── backtester.py           # The main application for running a historical simulation of the strategy.
├── live_bot.py             # The main application for running the bot live on a paper trading account.
├── .env                    # Stores secret API keys (not included in version control).
└── requirements.txt        # A list of all the Python libraries needed to run the project.

## How to Use This Project

Follow these steps to set up and run the bot.

**1. Set Up Your API Keys**
* You will need a free paper trading account from Alpaca and a free API key from MarketAux.
* Create a file named `.env` in the main project directory.
* Open the `.env` file and add your secret keys:
    ```
    API_KEY="your_alpaca_key"
    API_SECRET="your_alpaca_secret"
    MARKETAUX_API_KEY="your_marketaux_key"
    ```

**2. Install Dependencies**
* It is highly recommended to use a Python virtual environment to avoid conflicts.
* Install all the required libraries by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```

**3. Run the Backtester**
* To run a historical simulation of the strategy using the data in the `/data` folder, first ensure the data files have been generated by the scripts in the `/scripts` folder. Then, run:
    ```bash
    python backtester.py
    ```
* This will print a log of the simulated trades and display a chart of the portfolio's performance.

**4. Launch the Live Bot**
* To run the bot in live paper trading mode, where it will run continuously, execute:
    ```bash
    python live_bot.py
    ```
* You can monitor its activity in the terminal and see any trades it places in your Alpaca paper trading account. To stop the bot, press `Ctrl + C` in the terminal.

---

## Future Improvements

This project is a robust foundation that could be expanded in many ways:

* **Expand the Dataset:** The backtester's reliability is only as good as its historical data. A key next step would be to test the strategy on a much larger and more diverse historical news dataset covering several years.
* **Integrate a UK Broker:** The live trading module currently uses Alpaca, which is not ideal for UK stocks. The `executor.py` module could be swapped out for one that connects to a UK-friendly broker like Interactive Brokers.
* **Add More Strategy Rules:** The bot's decision engine is modular, making it easy to add more filters. A future version could incorporate a "choppiness" filter using the Average True Range (ATR) to prevent trading in sideways markets, a concept explored in my first project.
