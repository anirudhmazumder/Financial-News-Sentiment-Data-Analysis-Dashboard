<h1 align="center">
<b>Financial News Sentiment Analysis & Visualization</b>
</h1>
<h3 align="center">
<b>By: Anirudh Mazumder</b>
</h3>

## Project Overview
In this project I built a full-stack web application which scrapes financial news, perform sentiment analysis using a fine-tuned BERT model, computes a sentiment index, and visualizes the results on a web dashboard. My goal for this project was to provide insights into financial news sentiment trends over time and their potential impact on market behavior.

## Tech Stack

- Backend: Python, FastAPI

- Sentiment Analysis: Hugging Face Transformers (BERT), PyTorch

- Frontend: React, Tailwind CSS

- Data Visualization: Recharts

- Scraping: Requests, BeautifulSoup

## Features
### 1. News Scraping

Financial news headlines are scraped from [Finviz](https://finviz.com/news.ashx) using BeautifulSoup and Python Requests.

### 2. Sentiment Analysis

Developed a custom fine-tuned BERT model which is trained on the Kaggle [Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis). The algorithm analyzes each headline for sentiment (positive, negative, neutral) and computes confidence scores.

### 3. Sentiment Index

Sentiment scores are aggregated to produce a sentiment index over time.

### 4. Web Application

The web application was built using React and Tailwind CSS. It contains three main pages, a home page, headlines tab, and data analysis tab. The home page contains basic information about the platform. The headlines tab displays all the headlines with details including date, sentiment, and confidence. The data analysis tab contains three main pieces of information. First, a chart for sentiment index across days, a histogram of sentiment confidence scores, and a pie chart showing the distribution of sentiments. Finally, an API backend was built using Python's FastAPI library which provides sentiment data and daily sentiment index to the React frontend in real-time, enabling automated updates without manual pipeline runs.

## Citations

Real-time Financial News Headlines: https://finviz.com/news.ashx

Financial Sentiment Analysis Dataset: https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis