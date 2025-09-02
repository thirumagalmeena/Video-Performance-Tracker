import streamlit as st
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from scipy.stats import ttest_ind
from urllib.parse import urlparse, parse_qs
from src.api.youtube_fetcher import youtube

import re

from src.api.youtube_fetcher import (
    fetch_video_details, save_video_to_db,
    fetch_comments, save_comments_to_db,
    save_snapshot, search_videos_by_title,
    fetch_channel_videos, fetch_channel_details, save_channel_to_db
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

def extract_video_id(url_or_id):
    """Extract YouTube video ID from a URL or raw ID."""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
        if match:
            return match.group(1)
    elif len(url_or_id) == 11:  # already looks like video ID
        return url_or_id
    return None


def extract_channel_id(url_or_id):
    """
    Extract channel ID from URL, @handle, or direct ID.
    Returns UCxxxx channel ID if possible.
    """
    if url_or_id.startswith("UC"):  # already a channelId
        return url_or_id

    # /channel/UCxxxx
    match = re.search(r"channel/(UC[\w-]+)", url_or_id)
    if match:
        return match.group(1)

    # @handle → need search API
    if url_or_id.startswith("@"):
        request = youtube.search().list(
            part="snippet",
            q=url_or_id,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        if "items" in response and response["items"]:
            return response["items"][0]["snippet"]["channelId"]

    return None

# -------------------------------
# MAIN APP
# -------------------------------
st.set_page_config(page_title="YouTube Analytics Dashboard", page_icon="📊", layout="wide")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Home page
if st.session_state.page == 'home':
    st.title("📊 YouTube Analytics Dashboard")
    st.markdown("""
    Welcome to the YouTube Analytics Dashboard! Select an option below to get started:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎬 Video Performance")
        st.markdown("Analyze individual video performance, trends, and audience sentiment")
        if st.button("Explore Video Analytics", key="video_btn", use_container_width=True):
            st.session_state.page = 'video'
            st.rerun()
    
    with col2:
        st.subheader("📺 Channel Performance")
        st.markdown("Get insights into channel performance and compare videos")
        if st.button("Explore Channel Analytics", key="channel_btn", use_container_width=True):
            st.session_state.page = 'channel'
            st.rerun()
    
    st.divider()
    st.markdown("""
    **Features include:**
    - Video performance tracking over time
    - Comment sentiment analysis
    - View forecasting with Prophet
    - Statistical analysis and correlation
    - Channel performance metrics
    - Top video comparisons
    """)

# Video Performance Page
elif st.session_state.page == 'video':
    st.title("🎬 Video Performance Analysis")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    # Input section
    col1, col2 = st.columns([2, 1])
    with col1:
        url_or_id = st.text_input(
            "Enter YouTube Video URL or ID", 
            placeholder="https://youtu.be/dQw4w9WgXcQ or dQw4w9WgXcQ"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_query = st.button("🔍 Search YouTube by Title/Keyword")
    
    if search_query:
        with st.expander("Search YouTube Videos"):
            search_term = st.text_input("Enter search term")
            if search_term:
                results = search_videos_by_title(search_term)
                if results:
                    selected = st.selectbox(
                        "Select a video from search results:",
                        [f"{r['title']} ({r['channel']})" for r in results]
                    )
                    # Map back to video_id
                    selected_idx = [f"{r['title']} ({r['channel']})" for r in results].index(selected)
                    selected_video = results[selected_idx]
                    url_or_id = selected_video["video_id"]
                    st.success(f"Selected: {selected_video['title']}")
                else:
                    st.warning("No results found. Try another keyword.")
    
    if url_or_id:
        video_id = extract_video_id(url_or_id)
        
        if video_id:
            sub1, sub2, sub3, sub4 = st.tabs(["Overview", "Trends", "Forecast", "Analytics"])

            with sub1:  # Overview
                df_video = load_video_from_db(video_id)
                
                if not df_video.empty:
                    st.subheader(df_video.iloc[0]["title"])
                    st.write(f"**Channel:** {df_video.iloc[0]['channel']} | **Published:** {df_video.iloc[0]['published_date']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Views", f"{df_video.iloc[0]['views']:,}")
                    col2.metric("Likes", f"{df_video.iloc[0]['likes']:,}")
                    col3.metric("Comments", f"{df_video.iloc[0]['comments']:,}")
                    
                    # Calculate engagement rate
                    engagement_rate = ((df_video.iloc[0]['likes'] + df_video.iloc[0]['comments']) / 
                                      max(df_video.iloc[0]['views'], 1)) * 100
                    col4.metric("Engagement Rate", f"{engagement_rate:.2f}%")

                    st.success("✅ Loaded from database cache.")

                # Refresh button
                if st.button("🔄 Refresh Data from YouTube API", key="refresh_video"):
                    with st.spinner("Fetching latest data from YouTube..."):
                        details = fetch_video_details(video_id)
                        if details:
                            save_video_to_db(details)
                            save_snapshot(details)
                            
                            st.subheader(details["title"])
                            st.write(f"**Channel:** {details['channel']} | **Published:** {details['published_date']}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Views", f"{details['views']:,}")
                            col2.metric("Likes", f"{details['likes']:,}")
                            col3.metric("Comments", f"{details['comments']:,}")
                            
                            engagement_rate = ((details['likes'] + details['comments']) / 
                                              max(details['views'], 1)) * 100
                            col4.metric("Engagement Rate", f"{engagement_rate:.2f}%")

                            comments = fetch_comments(video_id)
                            save_comments_to_db(comments)
                            st.success(f"Fetched and saved {len(comments)} comments.")

                # Show sentiment if comments exist
                df_comments = load_comments_from_db(video_id)
                if not df_comments.empty:
                    st.subheader("Sentiment Analysis")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        sentiment_counts = df_comments["sentiment"].value_counts()
                        fig, ax = plt.subplots()
                        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                               startangle=90, colors=['#4CAF50', '#F44336', '#FFC107'])
                        ax.axis("equal")
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("#### Comment Sentiment Distribution")
                        for sentiment, count in sentiment_counts.items():
                            st.write(f"- **{sentiment.capitalize()}**: {count} comments ({count/len(df_comments)*100:.1f}%)")
                        
                        st.write("#### Sample Comments")
                        sample_comments = df_comments[["author", "text", "sentiment"]].head(5)
                        for _, comment in sample_comments.iterrows():
                            sentiment_color = {
                                "positive": "green",
                                "negative": "red",
                                "neutral": "orange"
                            }.get(comment["sentiment"], "gray")
                            
                            st.markdown(
                                f"<div style='border-left: 4px solid {sentiment_color}; padding-left: 10px; margin: 5px 0;'>"
                                f"<b>{comment['author']}</b> ({comment['sentiment']}):<br>{comment['text']}"
                                f"</div>", 
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No comments available. Click 'Refresh Data' to fetch comments.")

            with sub2:  # Trends
                st.subheader("Performance Trends Over Time")
                df_snapshots = load_snapshots(video_id)
                if not df_snapshots.empty:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df_snapshots["snapshot_date"]):
                        df_snapshots["snapshot_date"] = pd.to_datetime(df_snapshots["snapshot_date"])
                    
                    # Calculate daily changes
                    df_snapshots = df_snapshots.sort_values("snapshot_date")
                    df_snapshots["views_change"] = df_snapshots["views"].diff()
                    df_snapshots["likes_change"] = df_snapshots["likes"].diff()
                    df_snapshots["comments_change"] = df_snapshots["comments"].diff()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Cumulative Metrics")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(df_snapshots["snapshot_date"], df_snapshots["views"], label="Views", linewidth=2)
                        ax.plot(df_snapshots["snapshot_date"], df_snapshots["likes"], label="Likes", linewidth=2)
                        ax.plot(df_snapshots["snapshot_date"], df_snapshots["comments"], label="Comments", linewidth=2)
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Count")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    with col2:
                        st.write("#### Daily Changes")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(df_snapshots["snapshot_date"], df_snapshots["views_change"], 
                              alpha=0.7, label="Views Change")
                        ax.bar(df_snapshots["snapshot_date"], df_snapshots["likes_change"], 
                              alpha=0.7, label="Likes Change")
                        ax.bar(df_snapshots["snapshot_date"], df_snapshots["comments_change"], 
                              alpha=0.7, label="Comments Change")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Daily Change")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    # Summary statistics
                    st.write("#### Summary Statistics")
                    summary_cols = st.columns(4)
                    summary_cols[0].metric("Total Views", f"{df_snapshots['views'].iloc[-1]:,}")
                    summary_cols[1].metric("Total Likes", f"{df_snapshots['likes'].iloc[-1]:,}")
                    summary_cols[2].metric("Total Comments", f"{df_snapshots['comments'].iloc[-1]:,}")
                    summary_cols[3].metric("Days Tracked", len(df_snapshots))
                    
                else:
                    st.info("No snapshots yet. Fetch video data to create the first snapshot.")

            with sub3:  # Forecast
                st.subheader("Views Forecasting with Prophet")
                df_snapshots = load_snapshots(video_id)
                if df_snapshots.shape[0] >= 7:  # Prophet works better with more data
                    df_prophet = df_snapshots[["snapshot_date", "views"]].rename(
                        columns={"snapshot_date": "ds", "views": "y"}
                    )

                    with st.spinner("Training forecasting model..."):
                        model = Prophet(daily_seasonality=True, yearly_seasonality=False)
                        model.fit(df_prophet)

                        future = model.make_future_dataframe(periods=14)  # predict next 14 days
                        forecast = model.predict(future)

                    # Plot forecast
                    st.write("#### Forecasted Views")
                    fig1 = model.plot(forecast)
                    plt.title(f"Views Forecast for {df_video.iloc[0]['title'][:50]}...")
                    plt.xlabel("Date")
                    plt.ylabel("Views")
                    st.pyplot(fig1)

                    # Components (trend, seasonality)
                    st.write("#### Forecast Components")
                    fig2 = model.plot_components(forecast)
                    st.pyplot(fig2)
                    
                    # Show forecast values
                    st.write("#### Forecast Values")
                    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)
                    forecast_df.columns = ['Date', 'Predicted Views', 'Lower Bound', 'Upper Bound']
                    st.dataframe(forecast_df.style.format({
                        'Predicted Views': '{:,.0f}',
                        'Lower Bound': '{:,.0f}',
                        'Upper Bound': '{:,.0f}'
                    }))
                else:
                    st.info("Not enough snapshot data for forecasting. We need at least 7 days of data for reliable forecasts.")

            with sub4:  # Analytics
                st.subheader("Statistical Analysis")
                
                df_snapshots = load_snapshots(video_id)
                df_comments = load_comments_from_db(video_id)
                
                if not df_snapshots.empty:
                    # --- Correlation Heatmap ---
                    st.write("#### Correlation Between Metrics")
                    corr = df_snapshots[["views", "likes", "comments"]].corr()

                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", 
                               square=True, cbar_kws={"shrink": .8})
                    ax.set_title("Correlation Matrix")
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.write("**Interpretation:**")
                    st.write("- Values close to 1.0 indicate strong positive correlation")
                    st.write("- Values close to -1.0 indicate strong negative correlation")
                    st.write("- Values near 0 indicate little to no correlation")
                else:
                    st.info("Not enough snapshot data to compute correlations.")
                
                if not df_comments.empty:
                    # --- Hypothesis Test ---
                    st.write("#### Engagement Analysis by Sentiment")
                    
                    # Prepare data
                    df_comments["length"] = df_comments["text"].apply(len)
                    pos_lengths = df_comments[df_comments["sentiment"] == "positive"]["length"]
                    neg_lengths = df_comments[df_comments["sentiment"] == "negative"]["length"]
                    
                    if len(pos_lengths) > 5 and len(neg_lengths) > 5:
                        # Visualization
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sentiment_data = [pos_lengths, neg_lengths]
                            ax.boxplot(sentiment_data, labels=["Positive", "Negative"])
                            ax.set_ylabel("Comment Length (characters)")
                            ax.set_title("Comment Length by Sentiment")
                            st.pyplot(fig)
                        
                        with col2:
                            # T-test
                            t_stat, p_val = ttest_ind(pos_lengths, neg_lengths, equal_var=False)
                            
                            st.metric("T-statistic", f"{t_stat:.3f}")
                            st.metric("P-value", f"{p_val:.4f}")
                            
                            if p_val < 0.05:
                                st.success("✅ Statistically significant difference: Positive and negative comments have different engagement patterns.")
                                if t_stat > 0:
                                    st.write("Positive comments tend to be longer than negative comments.")
                                else:
                                    st.write("Negative comments tend to be longer than positive comments.")
                            else:
                                st.warning("❌ No significant difference found between positive and negative comment engagement.")
                    else:
                        st.info("Not enough positive/negative comments to run statistical analysis.")
                else:
                    st.info("No comments available for sentiment analysis.")
        else:
            st.error("Invalid YouTube URL or ID. Please check your input.")

# Channel Performance Page
elif st.session_state.page == 'channel':
    st.title("📺 Channel Performance Analysis")
    
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    channel_input = st.text_input(
        "Enter Channel URL, @handle, or ID", 
        placeholder="https://www.youtube.com/@channelname or @channelname or UCxxxxxx"
    )
    
    if channel_input:
        channel_id = extract_channel_id(channel_input)
        
        if channel_id:
            if st.button("Fetch Channel Data", key="fetch_channel"):
                with st.spinner("Fetching channel data..."):
                    channel_data = fetch_channel_details(channel_id)
                    
                    if channel_data:
                        save_channel_to_db(channel_data)

                        # --- Show channel overview ---
                        st.subheader(channel_data["title"])
                        st.write(channel_data["description"])
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Subscribers", f"{channel_data['subscribers']:,}")
                        col2.metric("Total Views", f"{channel_data['total_views']:,}")
                        col3.metric("Video Count", f"{channel_data['video_count']}")
                        
                        # Calculate average views per video
                        avg_views = channel_data['total_views'] / max(channel_data['video_count'], 1)
                        col4.metric("Avg Views/Video", f"{avg_views:,.0f}")

                        # --- Fetch channel videos ---
                        videos = fetch_channel_videos(channel_id, max_results=20)
                        if videos:
                            st.success(f"Fetched {len(videos)} videos from this channel")
                            df_videos = pd.DataFrame(videos)
                            
                            # Convert publishedAt to datetime
                            df_videos["publishedAt"] = pd.to_datetime(df_videos["publishedAt"])
                            
                            # Calculate engagement metrics
                            df_videos["engagement"] = (df_videos["likes"] + df_videos.get("comments", 0)) / df_videos["views"].replace(0, 1)
                            df_videos["engagement_rate"] = df_videos["engagement"] * 100
                            
                            # Display metrics in tabs
                            chan_tab1, chan_tab2, chan_tab3 = st.tabs(["Performance Over Time", "Top Videos", "Engagement Analysis"])
                            
                            with chan_tab1:
                                st.subheader("Performance Over Time")
                                
                                # Time series of views and likes
                                fig, ax = plt.subplots(figsize=(12, 6))
                                ax.scatter(df_videos["publishedAt"], df_videos["views"], 
                                          alpha=0.7, s=50, label="Views")
                                ax.scatter(df_videos["publishedAt"], df_videos["likes"] * 50, 
                                          alpha=0.7, s=50, label="Likes (scaled)")
                                ax.set_xlabel("Published Date")
                                ax.set_ylabel("Count")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                plt.xticks(rotation=45)
                                st.pyplot(fig)
                                
                                # Summary statistics
                                st.subheader("Channel Summary")
                                sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
                                sum_col1.metric("Total Videos", len(df_videos))
                                sum_col2.metric("Avg Views", f"{df_videos['views'].mean():,.0f}")
                                sum_col3.metric("Avg Likes", f"{df_videos['likes'].mean():,.0f}")
                                sum_col4.metric("Avg Engagement", f"{df_videos['engagement_rate'].mean():.2f}%")
                            
                            with chan_tab2:
                                st.subheader("Top Performing Videos")
                                
                                # Top 5 Videos by Views (one below another)
                                st.write("##### Top 5 Videos by Views")
                                top_views = df_videos.nlargest(5, "views")[["title", "views", "likes", "comments", "engagement_rate"]]
                                top_views["engagement_rate"] = top_views["engagement_rate"].round(2)
                                
                                # Create a horizontal bar chart for top videos by views
                                fig, ax = plt.subplots(figsize=(10, 6))
                                y_pos = range(len(top_views))
                                ax.barh(y_pos, top_views["views"], color='skyblue')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels([title[:40] + "..." if len(title) > 40 else title for title in top_views["title"]])
                                ax.invert_yaxis()  # highest at top
                                ax.set_xlabel('Views')
                                ax.set_title('Top 5 Videos by Views')
                                st.pyplot(fig)
                                
                                # Display the dataframe
                                st.dataframe(top_views.style.format({
                                    "views": "{:,.0f}",
                                    "likes": "{:,.0f}",
                                    "comments": "{:,.0f}",
                                    "engagement_rate": "{:.2f}%"
                                }))
                                
                                st.divider()
                                
                                # Top 5 Videos by Engagement (one below another)
                                st.write("##### Top 5 Videos by Engagement")
                                top_engaged = df_videos.nlargest(5, "engagement_rate")[["title", "views", "likes", "comments", "engagement_rate"]]
                                top_engaged["engagement_rate"] = top_engaged["engagement_rate"].round(2)
                                
                                # Create a horizontal bar chart for top videos by engagement
                                fig, ax = plt.subplots(figsize=(10, 6))
                                y_pos = range(len(top_engaged))
                                ax.barh(y_pos, top_engaged["engagement_rate"], color='orange')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels([title[:40] + "..." if len(title) > 40 else title for title in top_engaged["title"]])
                                ax.invert_yaxis()  # highest at top
                                ax.set_xlabel('Engagement Rate (%)')
                                ax.set_title('Top 5 Videos by Engagement Rate')
                                st.pyplot(fig)
                                
                                # Display the dataframe
                                st.dataframe(top_engaged.style.format({
                                    "views": "{:,.0f}",
                                    "likes": "{:,.0f}",
                                    "comments": "{:,.0f}",
                                    "engagement_rate": "{:.2f}%"
                                }))
                            
                            with chan_tab3:
                                st.subheader("Engagement Analysis")
                                
                                # Scatter plot of views vs engagement
                                fig, ax = plt.subplots(figsize=(10, 6))
                                scatter = ax.scatter(df_videos["views"], df_videos["engagement_rate"], 
                                                    alpha=0.7, s=60)
                                ax.set_xlabel("Views")
                                ax.set_ylabel("Engagement Rate (%)")
                                ax.set_title("Views vs Engagement Rate")
                                ax.grid(True, alpha=0.3)
                                
                                # Add trendline
                                z = np.polyfit(df_videos["views"], df_videos["engagement_rate"], 1)
                                p = np.poly1d(z)
                                ax.plot(df_videos["views"], p(df_videos["views"]), "r--", alpha=0.8)
                                
                                st.pyplot(fig)
                                
                                # Correlation analysis
                                st.write("##### Correlation Matrix")
                                corr = df_videos[["views", "likes", "comments", "engagement_rate"]].corr()
                                
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", 
                                           square=True, cbar_kws={"shrink": .8})
                                ax.set_title("Correlation Between Metrics")
                                st.pyplot(fig)
                                
                                st.write("**Insights:**")
                                st.write("- High correlation between views and likes indicates popular videos get more engagement")
                                st.write("- Engagement rate often decreases as views increase (common phenomenon)")
                        else:
                            st.warning("No videos found for this channel.")
                    else:
                        st.error("Could not fetch channel details. Please check the channel ID or URL.")
        else:
            st.error("Invalid channel identifier. Please use a channel URL, @handle, or channel ID.")