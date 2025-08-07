import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from io import StringIO

# Definition for the requests to extract the news data from finviz
url = "https://finviz.com/news.ashx"
req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
webpage = urlopen(req).read()
html = soup(webpage, "html.parser")

# Function to scrape news data from the HTML content
def scrape_news(html, idx):
    try:
        tables = pd.read_html(StringIO(str(html)))
        print(f"Found {len(tables)} tables.")
        if idx >= len(tables):
            print(f"Table index {idx} out of range.")
            return None
        news = tables[idx]
        print("Table preview:\n", news.head())
        if news.shape[1] == 3:
            news.columns = ["0", "Time", "Headlines"]
            news = news.drop(columns=["0"])
            news = news.set_index("Time")
            return news
        else:
            print(f"Table at index {idx} has {news.shape[1]} columns, expected 3.")
            return news
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to extract headlines from a specific table
def extract_headlines(table):
    table.columns = ["Unused", "Time", "Headline"]
    table = table.drop(columns=["Unused"])
    table = table.set_index("Time")

    return table

tables = pd.read_html(StringIO(str(html)))
print(f"Found {len(tables)} tables.")

# Extract and combine headlines from tables 3 and 4
dfs = []
for idx in [3, 4]:
    news_df = extract_headlines(tables[idx])
    dfs.append(news_df.reset_index())

combined_df = pd.concat(dfs, ignore_index=True)
print(combined_df.head())
combined_df.to_csv("news_headlines.csv", index=False)