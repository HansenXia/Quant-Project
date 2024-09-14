# Required Libraries
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from scipy.fft import fft

# Required NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')

# Web scraping dependencies
from urllib.parse import urlparse
import validators

# Sentiment analysis and NLP
from textblob import TextBlob

# Default Parameters
DEFAULT_TIME_WINDOW = 48  # Default time window for web scraping, in hours
CREDIBLE_DOMAINS = ["nytimes.com", "wsj.com", "bloomberg.com"]  # Add more trusted domains as needed

# --------------------------------------------
# Fetching News Articles for Sentiment Analysis
# --------------------------------------------
def fetch_news_articles(stock="AAPL", time_window=DEFAULT_TIME_WINDOW):
    """
    Function to scrape news articles related to the given stock (AAPL by default) from Google News.
    Filters out articles based on the time window provided and only keeps credible sources.
    
    Parameters:
    - stock: The stock symbol (default is "AAPL").
    - time_window: The number of hours to search through recent news (default 48 hours).

    Returns:
    - List of tuples containing the news title, URL, and publication date.
    """
    base_url = f"https://news.google.com/rss/search?q={stock}"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'xml')

    # Initialize list to store news data
    news_data = []
    
    # Iterate through all the news articles
    for item in soup.find_all('item'):
        pub_date = item.pubDate.text
        article_date = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
        
        # Filter articles based on time window
        if article_date >= datetime.now() - timedelta(hours=time_window):
            title = item.title.text
            link = item.link.text
            domain = urlparse(link).netloc
            
            # Skip low credibility websites
            if validators.domain(domain) and domain not in CREDIBLE_DOMAINS:
                continue  # Skips any source not in the trusted list
            
            news_data.append((title, link, article_date))  # Append the relevant news data

    return news_data

# ----------------------------------------
# Sentiment Analysis and Keyword Extraction
# ----------------------------------------
def extract_sentiment(news_data):
    """
    Function to analyze sentiment from the fetched news data using TextBlob.
    It also extracts the most frequent keywords from the news articles.
    
    Parameters:
    - news_data: List of news articles (title, link, publication date).

    Returns:
    - DataFrame of average sentiment over time.
    - DataFrame with keyword frequencies.
    """
    stop_words = set(stopwords.words('english'))  # List of common stopwords
    all_texts = " ".join([article[0] for article in news_data])  # Combine all news titles
    
    # Using CountVectorizer to find frequent keywords
    vectorizer = CountVectorizer(stop_words=stop_words)
    word_count = vectorizer.fit_transform([all_texts])
    keyword_freq = pd.DataFrame(word_count.toarray(), columns=vectorizer.get_feature_names_out())

    # List to store sentiment scores
    sentiment_scores = []
    for title, link, date in news_data:
        blob = TextBlob(title)  # Apply TextBlob for sentiment analysis
        sentiment_scores.append(blob.sentiment.polarity)  # Store sentiment score for each article

    # Create a DataFrame with the sentiment scores and dates
    sentiment_df = pd.DataFrame({
        'Title': [data[0] for data in news_data],
        'Sentiment': sentiment_scores,
        'Date': [data[2] for data in news_data]
    })
    
    # Aggregate sentiment scores by date
    avg_sentiment = sentiment_df.groupby('Date').mean().reset_index()
    
    return avg_sentiment, keyword_freq

# ------------------------------------------
# Stock Price Prediction using Brownian Motion
# ------------------------------------------
def brownian_motion_simulation(S0, mu, sigma, T=1, N=1000):
    """
    Function to simulate stock price using a Brownian motion model.
    
    Parameters:
    - S0: Initial stock price (starting point).
    - mu: Mean (drift rate) calculated from past data.
    - sigma: Volatility calculated from past stock price data.
    - T: Time horizon for the simulation (default 1 day).
    - N: Number of steps in the simulation (default 1000).

    Returns:
    - Simulated stock price path using Brownian motion.
    """
    dt = T / N  # Time step
    W = np.random.standard_normal(size=N)  # Generate standard normal random numbers
    W = np.cumsum(W) * np.sqrt(dt)  # Compute the Wiener process
    t = np.linspace(0, T, N)  # Create time vector
    X = (mu - 0.5 * sigma**2) * t + sigma * W  # Apply the Brownian motion formula
    return S0 * np.exp(X)  # Return the simulated price

# --------------------------------------
# Fourier Transform for Time Series Analysis
# --------------------------------------
def perform_fourier_transform(stock_data):
    """
    Function to perform Fourier Transform on stock price data to identify patterns.
    
    Parameters:
    - stock_data: Stock price data (pandas DataFrame).

    Returns:
    - Fourier transformed data.
    """
    close_prices = stock_data['Close'].values  # Extract closing prices
    transformed = fft(close_prices)  # Apply Fourier Transform
    return transformed

# ----------------------------
# Fetch Stock Data from Yahoo Finance
# ----------------------------
def fetch_stock_data(stock="AAPL", days=7):
    """
    Function to fetch stock price data from Yahoo Finance.
    
    Parameters:
    - stock: The stock symbol (default "AAPL").
    - days: The number of past days to fetch data for (default 7 days).

    Returns:
    - DataFrame containing stock data (Open, Close, High, Low, etc.).
    """
    stock = yf.Ticker(stock)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    stock_data = stock.history(start=start_date, end=end_date)
    return stock_data

# -----------------------------------------
# Main Prediction and Plotting Function
# -----------------------------------------
def predict_stock_price(stock="AAPL"):
    """
    Main function to predict stock price fluctuations using:
    - Sentiment Analysis
    - Brownian Motion
    - Fourier Transform
    And plot the combined results.
    
    Parameters:
    - stock: The stock symbol to predict (default "AAPL").
    """
    # Step 1: Scrape and Analyze Sentiment
    news_data = fetch_news_articles(stock)
    sentiment_data, keyword_freq = extract_sentiment(news_data)

    # Step 2: Fetch Stock Data and Perform Fourier Transform
    stock_data = fetch_stock_data(stock)
    fourier_transformed = perform_fourier_transform(stock_data)

    # Step 3: Brownian Motion Simulation
    S0 = stock_data['Close'].iloc[-1]  # Last closing price
    mu = stock_data['Close'].pct_change().mean()  # Average daily return
    sigma = stock_data['Close'].pct_change().std()  # Volatility (standard deviation)
    bm_simulation = brownian_motion_simulation(S0, mu, sigma)

    # Step 4: Combine Results into One Plot
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Sentiment over Time
    ax[0].plot(sentiment_data['Date'], sentiment_data['Sentiment'], label='Sentiment Score', color='blue')
    ax[0].set_title('Sentiment Score Over Time')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Sentiment')
    ax[0].legend()

    # Plot 2: Fourier Transform of Stock Prices
    ax[1].plot(np.abs(fourier_transformed), label='Fourier Transform', color='green')
    ax[1].set_title('Fourier Transform of Stock Prices')
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()

    # Plot 3: Brownian Motion Simulation for Price
    ax[2].plot(bm_simulation, label='Simulated Stock Price', color='red')
    ax[2].set_title(f'Brownian Motion Simulation of {stock} Stock Price')
    ax[2].set_xlabel('Time Steps')
    ax[2].set_ylabel('Price')
    ax[2].legend()

    plt.tight_layout()
    plt.show()

# -------------------------
# Running the Prediction
# -------------------------
predict_stock_price("AAPL")
