from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

class SentimentAnalysisPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        print(f"Attempting to load the model and tokenizer from {self.model_path}")
        
        abs_model_path = os.path.abspath(self.model_path)
        print(f"Absolute path: {abs_model_path}")

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"Model directory not found: {abs_model_path}")

        model = AutoModelForSequenceClassification.from_pretrained(
            abs_model_path, 
            local_files_only=True,
            trust_remote_code=False
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            abs_model_path, 
            local_files_only=True,
            trust_remote_code=False
        )

        print("The model and the tokenizer were loaded successfully.")

        return model, tokenizer

    def predict_sentiment(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()

        labels = ['Negative', 'Neutral', 'Positive']

        final_prediction = {
            'text': text,
            'sentiment': labels[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                'Negative': round(probabilities[0][0].item(), 4),
                'Neutral': round(probabilities[0][1].item(), 4),
                'Positive': round(probabilities[0][2].item(), 4)
            },
            'predicted_class': predicted_class
        }

        return final_prediction

class NewsDataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data
        

sentiment_analyzer = SentimentAnalysisPredictor("../Sentiment_Analysis_Model/saved_sentiment_model")
news_loader = NewsDataLoader("../News_Data_Extraction/news_headlines.csv")
news_data = news_loader.load_data()
final_output_df = pd.DataFrame(columns=['Time', 'Headline', 'Sentiment', 'Confidence', 'Negative_Probability', 'Neutral_Probability', 'Positive_Probability', 'Predicted_Class'])

for index, row in news_data.iterrows():
    financial_news_headline = row['Headline']
    prediction = sentiment_analyzer.predict_sentiment(financial_news_headline)
    print(f"Text: {financial_news_headline}\nPrediction: {prediction}\n")

    # Append the prediction results to the final output DataFrame
    final_output_df = final_output_df._append({
        'Time': row['Time'],
        'Headline': financial_news_headline,
        'Sentiment': prediction['sentiment'],
        'Confidence': prediction['confidence'],
        'Negative_Probability': prediction['probabilities']['Negative'],
        'Neutral_Probability': prediction['probabilities']['Neutral'],
        'Positive_Probability': prediction['probabilities']['Positive'],
        'Predicted_Class': prediction['predicted_class']
    }, ignore_index=True)

final_output_df.to_csv("Finance_Sentiment_Analysis.csv")