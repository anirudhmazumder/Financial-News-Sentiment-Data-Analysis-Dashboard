import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import Request, urlopen
from io import StringIO
from datetime import datetime, date

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

# Function to add date to timestamps
def add_date_to_timestamps(time_str, current_date=None):
    if current_date is None:
        current_date = date.today()
    
    # Handle different time formats
    time_str = str(time_str).strip()
    
    # If it's already a full datetime, return as is
    if len(time_str) > 8 and ('/' in time_str or '-' in time_str):
        return time_str
    
    # Check if it's a date format like "Aug-07", "Jan-15", etc.
    if any(month in time_str for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
        current_year = current_date.year
        month_day = time_str.replace('-', ' ')
        full_date_str = f"{month_day} {current_year}"
        parsed_date = datetime.strptime(full_date_str, "%b %d %Y")
        return parsed_date.strftime("%Y-%m-%d")
    
    # Check if it's just a time (like "01:58AM", "12:32AM")
    if any(indicator in time_str.upper() for indicator in ['AM', 'PM']) or ':' in time_str:
        formatted_datetime = f"{current_date.strftime('%Y-%m-%d')} {time_str}"
        return formatted_datetime
    
    return time_str

# Function to extract headlines from a specific table
def extract_headlines(table):
    table.columns = ["Unused", "Time", "Headline"]
    table = table.drop(columns=["Unused"])

    table['Time'] = table['Time'].apply(add_date_to_timestamps)

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

if pd.api.types.is_datetime64_any_dtype(combined_df['Time']):
    combined_df = combined_df.sort_values('Time', ascending=False)

combined_df.to_csv("news_headlines.csv", index=False)