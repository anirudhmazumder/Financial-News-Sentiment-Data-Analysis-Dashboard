import pandas as pd
from datetime import datetime, date

class SentimentIndexCalculator:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path

    def parse_timestamps(self, time_str):
        time_obj = pd.to_datetime(time_str, errors='coerce')
        if pd.notna(time_obj):
            return time_obj.date()
        return pd.to_datetime(time_str, errors='coerce')

    def load_financial_data(self):
        finance_data = pd.read_csv(self.data_path)
        finance_data['Predicted_Class'] = finance_data['Predicted_Class'].map({
            0: 'Negative',
            1: 'Neutral',
            2: 'Positive'
        })
        finance_data['Time'] = finance_data['Time'].apply(self.parse_timestamps)
        return finance_data

    def daily_sentiment_index_calculation(self):
        finance_data = self.load_financial_data()
        sentiment_index_values = {
            'Negative': -1,
            'Neutral': 0,
            'Positive': 1
        }
        finance_data['Sentiment_Score'] = finance_data['Predicted_Class'].map(sentiment_index_values) * finance_data['Confidence']
        
        daily_sentiment_index = finance_data.groupby('Time')['Sentiment_Score'].mean().mul(100).reset_index()
        daily_sentiment_index.rename(columns={'Sentiment_Score': 'Daily_Sentiment_Index'}, inplace=True)
        daily_sentiment_index['Headlines_Processed'] = finance_data.groupby('Time').size().values

        daily_sentiment_index.to_csv(self.output_path, index=False)
        
        return daily_sentiment_index