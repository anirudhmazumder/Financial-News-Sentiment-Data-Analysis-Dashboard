from Data_Analysis_Components import NewsDataExtractor, SentimentAnalyzer, SentimentIndexCalculator, MarketPredictor, ConvertToJSON

def run_pipeline():
    newsDataExtractor = NewsDataExtractor.NewsDataExtractor("Financial_News_Data/Financial_News_Headlines.csv")
    news = newsDataExtractor.extract_news_data()

    financialNewsSentimentAnalysis = SentimentAnalyzer.SentimentAnalyzer("Sentiment_Analysis_Model/saved_sentiment_model", "Financial_News_Data/Financial_News_Headlines.csv", "Financial_News_Data/Finance_Sentiment_Analysis.csv")
    sentiment_results = financialNewsSentimentAnalysis.extract_sentiment_for_all_headlines()

    json_converter = ConvertToJSON.ConvertToJSON(sentiment_results)
    json_converter.convert("Financial_News_Data/Sentiment_Analysis_Results.json")

    sentimentIndexCalculation = SentimentIndexCalculator.SentimentIndexCalculator("Financial_News_Data/Finance_Sentiment_Analysis.csv", "Financial_News_Data/Daily_Sentiment_Index.csv")
    daily_sentiment_index = sentimentIndexCalculation.daily_sentiment_index_calculation()

    json_converter = ConvertToJSON.ConvertToJSON(daily_sentiment_index)
    json_converter.convert("Financial_News_Data/Daily_Sentiment_Index.json")

run_pipeline()