from googleapiclient.discovery import build
from src.database.db_handler import get_connection, create_tables
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

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

def save_snapshot(video_data):
    """Save daily snapshot of views/likes/comments."""
    conn = get_connection()
    cursor = conn.cursor()
    today = datetime.date.today().isoformat()

    cursor.execute("""
    INSERT OR REPLACE INTO video_snapshots (video_id, snapshot_date, views, likes, comments)
    VALUES (?, ?, ?, ?, ?)
    """, (
        video_data["video_id"],
        today,
        video_data["views"],
        video_data["likes"],
        video_data["comments"]
    ))
    conn.commit()
    conn.close()
    print(f"Snapshot saved for {video_data['video_id']} on {today}")

def search_videos_by_title(query, max_results=5):
    """Search YouTube for videos by title/keyword."""
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    results = []
    for item in response.get("items", []):
        if item["id"]["kind"] == "youtube#video":  # safeguard
            results.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "publishedAt": item["snippet"]["publishedAt"]
            })
    return results
 
def fetch_channel_videos(channel_id, max_results=10):
    """Fetch recent videos from a channel along with stats."""
    request = youtube.search().list(
        part="snippet",
        channelId=channel_id,
        maxResults=max_results,
        order="date",
        type="video"
    )
    response = request.execute()

    videos = []
    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]

    if not video_ids:
        return videos

    # Fetch statistics for these videos
    stats_request = youtube.videos().list(
        part="statistics,snippet",
        id=",".join(video_ids)
    )
    stats_response = stats_request.execute()

    for item in stats_response.get("items", []):
        snippet = item["snippet"]
        stats = item["statistics"]
        videos.append({
            "video_id": item["id"],
            "title": snippet["title"],
            "publishedAt": snippet["publishedAt"],
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)),
            "comments": int(stats.get("commentCount", 0))
        })

    return videos


def fetch_channel_details(channel_id):
    """Fetch channel metadata + stats."""
    request = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    )
    response = request.execute()

    if "items" not in response or not response["items"]:
        print(f"[ERROR] No channel found for ID: {channel_id}")
        return None

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    return {
        "channel_id": channel_id,
        "title": snippet["title"],
        "description": snippet.get("description", ""),
        "subscribers": int(stats.get("subscriberCount", 0)),
        "total_views": int(stats.get("viewCount", 0)),
        "video_count": int(stats.get("videoCount", 0))
    }



def save_channel_to_db(channel_data):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO channels (channel_id, title, description, subscribers, total_views, video_count)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        channel_data["channel_id"],
        channel_data["title"],
        channel_data["description"],
        channel_data["subscribers"],
        channel_data["total_views"],
        channel_data["video_count"]
    ))
    conn.commit()
    conn.close()
