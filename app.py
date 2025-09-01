import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from src.api.youtube_fetcher import (
    fetch_video_details,
    save_video_to_db,
    fetch_comments,
    save_comments_to_db,
    save_snapshot,
)
from src.database.db_handler import create_tables

# --- Ensure tables exist ---
create_tables()

DB_PATH = "db/youtube.db"

# -------------------------------
# DB helper functions
# -------------------------------
def load_video_from_db(video_id):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM videos WHERE video_id = ?"
    df = pd.read_sql(query, conn, params=(video_id,))
    conn.close()
    return df

def load_comments_from_db(video_id):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM comments WHERE video_id = ?"
    df = pd.read_sql(query, conn, params=(video_id,))
    conn.close()
    return df

def load_snapshots(video_id):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM video_snapshots WHERE video_id = ? ORDER BY snapshot_date"
    df = pd.read_sql(query, conn, params=(video_id,))
    conn.close()
    return df

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📊 YouTube Video Performance Tracker")

video_id = st.text_input("Enter YouTube Video ID (e.g., dQw4w9WgXcQ)")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trends","Forecast","Analytics"])

# -------------------------------
# Tab 1: Overview
# -------------------------------
with tab1:
    if st.button("Fetch Video Data"):
        if video_id:
            # Step 1: fetch video details
            details = fetch_video_details(video_id)
            if details:
                save_video_to_db(details)
                save_snapshot(details)  # ✅ save daily snapshot

                # Show video summary
                st.subheader(details["title"])
                st.write(f"Channel: {details['channel']} | Published: {details['published_date']}")
                col1, col2, col3 = st.columns(3)
                col1.metric("Views", f"{details['views']:,}")
                col2.metric("Likes", f"{details['likes']:,}")
                col3.metric("Comments", f"{details['comments']:,}")

                # Step 2: fetch comments + save
                comments = fetch_comments(video_id)
                save_comments_to_db(comments)
                st.success(f"Fetched and saved {len(comments)} comments.")

                # Step 3: sentiment distribution
                df_comments = load_comments_from_db(video_id)
                if not df_comments.empty:
                    st.write("### Sentiment Analysis")
                    sentiment_counts = df_comments["sentiment"].value_counts()

                    fig, ax = plt.subplots()
                    ax.pie(
                        sentiment_counts,
                        labels=sentiment_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                    st.write("### Sample Comments")
                    st.dataframe(df_comments[["author", "text", "sentiment"]].head(10))
            else:
                st.error("Could not fetch video details. Check the video ID or API quota.")

# -------------------------------
# Tab 2: Trends
# -------------------------------
with tab2:
    if video_id:
        st.write("### Trends over time")
        df_snapshots = load_snapshots(video_id)
        if not df_snapshots.empty:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(df_snapshots["snapshot_date"], df_snapshots["views"], label="Views")
            ax.plot(df_snapshots["snapshot_date"], df_snapshots["likes"], label="Likes")
            ax.plot(df_snapshots["snapshot_date"], df_snapshots["comments"], label="Comments")
            ax.set_xlabel("Date")
            ax.set_ylabel("Count")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No snapshots yet. Fetch video data to create the first snapshot.")

# -------------------------------
# Tab 3: Forecast (Prophet)
# -------------------------------

with tab3:
    if video_id:
        st.write("### Forecasting Views with Prophet")
        df_snapshots = load_snapshots(video_id)
        if df_snapshots.shape[0] >= 2:  # Prophet needs at least 2 data points
            df_prophet = df_snapshots[["snapshot_date", "views"]].rename(
                columns={"snapshot_date": "ds", "views": "y"}
            )

            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=7)  # predict next 7 days
            forecast = model.predict(future)

            # Plot forecast
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Components (trend, seasonality)
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
        else:
            st.info("Not enough snapshot data for forecasting. Try collecting stats for multiple days.")

# -------------------------------
# Tab 4: Analytics
# -------------------------------
with tab4:
    if video_id:
        st.write("### Statistical Analysis")
        
        # --- Correlation Heatmap ---
        st.subheader("Correlation between Views, Likes, Comments")
        df_snapshots = load_snapshots(video_id)
        if not df_snapshots.empty:
            corr = df_snapshots[["views", "likes", "comments"]].corr()

            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough snapshot data to compute correlations.")

        # --- Hypothesis Test ---
        st.subheader("Do positive-sentiment videos get more comments?")
        df_comments = load_comments_from_db(video_id)
        if not df_comments.empty:
            # Group comments by sentiment
            pos_comments = df_comments[df_comments["sentiment"] == "positive"]["text"].count()
            neg_comments = df_comments[df_comments["sentiment"] == "negative"]["text"].count()

            # To test statistically, simulate by using comment lengths as a proxy for engagement
            df_comments["length"] = df_comments["text"].apply(len)
            pos_lengths = df_comments[df_comments["sentiment"] == "positive"]["length"]
            neg_lengths = df_comments[df_comments["sentiment"] == "negative"]["length"]

            if len(pos_lengths) > 1 and len(neg_lengths) > 1:
                t_stat, p_val = ttest_ind(pos_lengths, neg_lengths, equal_var=False)
                st.write(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.4f}")
                if p_val < 0.05:
                    st.success("✅ Statistically significant: Positive sentiment comments differ from negative ones in length/engagement.")
                else:
                    st.warning("❌ No significant difference found between positive and negative comment engagement.")
            else:
                st.info("Not enough positive/negative comments to run hypothesis test.")
        else:
            st.info("No comments available for sentiment analysis.")