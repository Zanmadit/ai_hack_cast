import requests
import feedparser
import pandas as pd
import datetime
from urllib.parse import quote
import time

categories = ["cs.AI", "cs.LG", "stat.ML", "cs.CL", "cs.CV"]

start_date = datetime.date.today() - datetime.timedelta(days=5 * 365)
end_date = datetime.date.today()

all_entries = []

for cat in categories:
    print(f"Category {cat}")
    start = 0
    batch_size = 100
    while True:
        query = quote(f"cat:{cat}")
        url = (
            f"https://export.arxiv.org/api/query?"
            f"search_query={query}&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        feed = feedparser.parse(requests.get(url).text)
        entries = feed.entries
        if not entries:
            break 
        
        for e in entries:
            pub_date = datetime.datetime.strptime(
                e.published.split("T")[0], "%Y-%m-%d"
            ).date()
            if pub_date < start_date:
                break  
            all_entries.append({
                "title": e.title,
                "published": pub_date,
                "category": cat,
                "authors": ", ".join([a.name for a in e.authors]),
                "summary": e.summary,
                "url": e.link
            })

        start += batch_size
        print(f"  Downloaded {len(entries)} records(total {len(all_entries)})")
        time.sleep(3)  

print(f"\n Total {len(all_entries)} articles")

df = pd.DataFrame(all_entries)
df.to_csv("data/arxiv_data.csv", index=False)
