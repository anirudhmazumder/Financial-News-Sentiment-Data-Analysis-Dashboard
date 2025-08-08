import pandas as pd

def loading_financial_data():
    finance_data = pd.read_csv("../Financial_Data_Sentiment_Analysis/Finance_Sentiment_Analysis.csv")
    finance_data['Predicted_Class'] = finance_data['Predicted_Class'].map({
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    })
    return finance_data

def sentiment_index_calculation(finance_data):
    sentiment_index_values = {
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1
    }
    finance_data['Sentiment_Score'] = finance_data['Predicted_Class'].map(sentiment_index_values) * finance_data['Confidence']
    daily_sentiment_index = finance_data.groupby('Time').agg(
        sentiment_score=('Sentiment_Score', 'mean'),
        headline_count=('Sentiment_Score', 'count')
    )
    return daily_sentiment_index * 100


finance_data = loading_financial_data()
daily_sentiment_index = sentiment_index_calculation(finance_data)
print(daily_sentiment_index.head())