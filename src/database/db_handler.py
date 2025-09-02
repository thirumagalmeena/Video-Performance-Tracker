import sqlite3

DB_PATH = "db/youtube.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        title TEXT,
        channel TEXT,
        published_date TEXT,
        views INTEGER,
        likes INTEGER,
        comments INTEGER
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        comment_id TEXT PRIMARY KEY,
        video_id TEXT,
        author TEXT,
        text TEXT,
        sentiment TEXT,
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_snapshots (
        video_id TEXT,
        snapshot_date TEXT,
        views INTEGER,
        likes INTEGER,
        comments INTEGER,
        PRIMARY KEY(video_id, snapshot_date),
        FOREIGN KEY(video_id) REFERENCES videos(video_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS channels (
        channel_id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        subscribers INTEGER,
        total_views INTEGER,
        video_count INTEGER
    )
    """)
    conn.commit()
    conn.close()

def fetch_data():
    conn = get_connection()
    cursor = conn.cursor()

    print("\nVideos:")
    for row in cursor.execute("SELECT * FROM videos"):
        print(row)

    print("\nComments:")
    for row in cursor.execute("SELECT * FROM comments"):
        print(row)

    conn.close()

def clear_tables():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM videos")
    cursor.execute("DELETE FROM comments")
    conn.commit()
    conn.close()
    print("All tables cleared!")

"""
if __name__ == "__main__":
    create_tables()
    fetch_data()
    clear_tables()
"""