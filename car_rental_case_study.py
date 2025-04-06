import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
import re
from collections import Counter


def scrape_reviews():
    urls = [
        "https://www.trustpilot.com/review/centauro.net",
        "https://centauro.net/en/reviews/italy/"
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    reviews = []

    for url in urls:
        try:
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, 'html.parser')
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50:
                    reviews.append(text)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return reviews[:200]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity


def extract_and_prepare_data():
    reviews = scrape_reviews()
    df = pd.DataFrame(reviews, columns=['review'])
    df['cleaned'] = df['review'].apply(clean_text)
    df['sentiment'] = df['cleaned'].apply(analyze_sentiment)
    return df


def plot_sentiment_distribution(df):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['sentiment'], bins=30, kde=True, color='skyblue')
    plt.title('Customer Sentiment Distribution')
    plt.xlabel('Sentiment Score (-1 = Negative, +1 = Positive)')
    plt.ylabel('Review Count')
    plt.axvline(df['sentiment'].mean(), color='red', linestyle='--', label='Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/sentiment_distribution.png")
    plt.close()


def plot_wordcloud(df):
    all_text = ' '.join(df['cleaned'])
    wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(15, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Common Words in All Reviews')
    plt.tight_layout()
    plt.savefig("outputs/wordcloud_all_reviews.png")
    plt.close()


def plot_top_negative_words(df):
    neg_reviews = df[df['sentiment'] < 0]
    words = ' '.join(neg_reviews['cleaned']).split()
    common = Counter([w for w in words if w not in ['the', 'and', 'to', 'a', 'of', 'in', 'for', 'is', 'it']])
    top_words = pd.DataFrame(common.most_common(10), columns=['word', 'count'])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_words, x='count', y='word', palette='Reds_r')
    plt.title('Top Complaints in Negative Reviews')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig("outputs/top_negative_words.png")
    plt.close()


def run_pipeline():
    df = extract_and_prepare_data()
    plot_sentiment_distribution(df)
    plot_wordcloud(df)
    plot_top_negative_words(df)


if __name__ == "__main__":
    run_pipeline()
