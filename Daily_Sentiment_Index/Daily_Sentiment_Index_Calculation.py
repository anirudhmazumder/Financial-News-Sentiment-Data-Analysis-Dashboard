import pandas as pd
from datetime import datetime, date

def parse_timestamps(time_str):
    time_obj = pd.to_datetime(time_str, errors='coerce')
    if pd.notna(time_obj):
        return time_obj.date()
    
    return pd.to_datetime(time_str, errors='coerce')

def loading_financial_data():
    finance_data = pd.read_csv("../Financial_Data_Sentiment_Analysis/Finance_Sentiment_Analysis.csv")
    finance_data['Predicted_Class'] = finance_data['Predicted_Class'].map({
        0: 'Negative',
        1: 'Neutral',
        2: 'Positive'
    })
    finance_data['Time'] = finance_data['Time'].apply(parse_timestamps)
    return finance_data

def sentiment_index_calculation(finance_data):
    sentiment_index_values = {
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1
    }
    finance_data['Sentiment_Score'] = finance_data['Predicted_Class'].map(sentiment_index_values) * finance_data['Confidence']
    
    daily_sentiment_index = finance_data.groupby('Time')['Sentiment_Score'].mean().mul(100).reset_index()
    daily_sentiment_index.rename(columns={'Sentiment_Score': 'Daily_Sentiment_Index'}, inplace=True)
    daily_sentiment_index['Headlines_Processed'] = finance_data.groupby('Time').size().values
    
    return daily_sentiment_index

finance_data = loading_financial_data()
daily_sentiment_index = sentiment_index_calculation(finance_data)
print(daily_sentiment_index)