from googleapiclient.discovery import build
from src.database.db_handler import get_connection, create_tables
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

# Initialize NLTK sentiment analyzer
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def fetch_video_details(video_id):
    """Fetch details of a single YouTube video."""
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()

    if not response["items"]:
        print("No video found.")
        return

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    data = {
        "video_id": video_id,
        "title": snippet["title"],
        "channel": snippet["channelTitle"],
        "published_date": snippet["publishedAt"],
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0))
    }
    return data

def save_video_to_db(video_data):
    """Insert video data into SQLite DB."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO videos (video_id, title, channel, published_date, views, likes, comments)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        video_data["video_id"],
        video_data["title"],
        video_data["channel"],
        video_data["published_date"],
        video_data["views"],
        video_data["likes"],
        video_data["comments"]
    ))
    conn.commit()
    conn.close()
    print(f"Saved video {video_data['title']} to DB!")

def fetch_comments(video_id, max_results=50):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    comments_data = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]
        text = comment["textDisplay"]
        
        # Run sentiment analysis
        sentiment_score = sia.polarity_scores(text)
        if sentiment_score["compound"] >= 0.05:
            sentiment = "positive"
        elif sentiment_score["compound"] <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        comments_data.append((
            item["id"],
            video_id,
            comment["authorDisplayName"],
            text,
            sentiment
        ))

    return comments_data

def save_comments_to_db(comments):
    conn = get_connection()
    cur = conn.cursor()

    cur.executemany("""
        INSERT OR IGNORE INTO comments 
        (comment_id, video_id, author, text, sentiment)
        VALUES (?, ?, ?, ?, ?)
    """, comments)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    video_id = "dQw4w9WgXcQ"  # Example video
    create_tables()
    vid_id = "dQw4w9WgXcQ"  # test with any YouTube video ID
    video_data = fetch_video_details(vid_id)
    if video_data:
        save_video_to_db(video_data)
    comments = fetch_comments(video_id)
    save_comments_to_db(comments)
    print(f"Saved {len(comments)} comments with sentiment into DB!")