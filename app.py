import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Music Sentiment Intelligence", layout="wide", initial_sidebar_state="expanded")

# 1. Load Data
@st.cache_data
def load_data():
    return pd.read_csv("music_sentiment_intelligence.csv")

df = load_data()

# 2. Sidebar Filters
st.sidebar.title("🎛 Catalog Filters")
artists = st.sidebar.multiselect(
    "Select Artist(s)",
    options=df["artist_name"].unique(),
    default=df["artist_name"].unique()
)

min_year, max_year = int(df["release_year"].min()), int(df["release_year"].max())
year_range = st.sidebar.slider("Release Year Range", min_year, max_year, (min_year, max_year))

# Filter dataframe
filtered_df = df[
    (df["artist_name"].isin(artists)) &
    (df["release_year"].between(year_range[0], year_range[1]))
]

# 3. Main Dashboard Header
st.title("🎵 Music & Lyric Sentiment Intelligence")
st.caption("A comparative NLP and acoustic feature analysis: Taylor Swift vs. Lana Del Rey")
st.markdown("---")

# 4. KPI Top Band
avg_valence = filtered_df["valence"].mean() if not filtered_df.empty else 0.0
avg_sentiment = filtered_df["vader_compound"].mean() if not filtered_df.empty else 0.0
avg_melancholy = filtered_df["melancholy_index"].mean() if not filtered_df.empty else 0.0
trojan_count = len(filtered_df[(filtered_df["valence"] >= 0.50) & (filtered_df["vader_compound"] < 0.0)])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Audio Valence", f"{avg_valence:.2f}", help="Spotify acoustic brightness (0-1)")
col2.metric("Avg Lyrical Mood", f"{avg_sentiment:+.2f}", help="VADER Compound sentiment (-1 to +1)")
col3.metric("Melancholy Index", f"{avg_melancholy:.2f}", help="Composite measure of sonic & lyrical somberness")
col4.metric("Deceptive Euphoria Tracks", f"{trojan_count}", help="Upbeat sound (Valence ≥ 0.5) + Sad lyrics (VADER < 0)")

st.markdown("---")

# 5. Visualizations Row
col_left, col_right = st.columns([1.2, 1])

color_map = {"Taylor Swift": "#e056fd", "Lana Del Rey": "#0984e3"}

with col_left:
    st.subheader("Emotional Quadrant Matrix")
    
    fig_scatter = px.scatter(
        filtered_df,
        x="vader_compound",
        y="valence",
        color="artist_name",
        hover_data=["track_name", "album_name", "sentiment_divergence"],
        color_discrete_map=color_map,
        labels={"vader_compound": "Lyrical Sentiment (VADER)", "valence": "Audio Valence"},
        height=480
    )
    
    fig_scatter.add_vline(x=0.0, line_dash="dash", line_color="gray", opacity=0.7)
    fig_scatter.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)
    
    fig_scatter.add_annotation(x=-0.8, y=0.95, text="II: Deceptive Euphoria (Upbeat & Sad)", showarrow=False, font=dict(color="#d63031", size=10))
    fig_scatter.add_annotation(x=0.6, y=0.95, text="I: Unified Optimism", showarrow=False, font=dict(color="#27ae60", size=10))
    fig_scatter.add_annotation(x=-0.8, y=0.05, text="III: Pure Melancholia", showarrow=False, font=dict(color="#2c3e50", size=10))
    fig_scatter.add_annotation(x=0.6, y=0.05, text="IV: Ambient Reflection", showarrow=False, font=dict(color="#e67e22", size=10))
    
    fig_scatter.update_layout(xaxis=dict(range=[-1.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]))
    st.plotly_chart(fig_scatter, width="stretch")

with col_right:
    st.subheader("Career Melancholy Trajectory")
    
    trend_df = filtered_df.groupby(["release_year", "artist_name"])["melancholy_index"].mean().reset_index()
    
    fig_line = px.line(
        trend_df,
        x="release_year",
        y="melancholy_index",
        color="artist_name",
        markers=True,
        color_discrete_map=color_map,
        labels={"release_year": "Release Year", "melancholy_index": "Avg Melancholy Index"},
        height=480
    )
    fig_line.update_layout(yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_line, width="stretch")

# 6. Deep Dive Data Table
st.subheader("Detailed Track Decomposition")
st.dataframe(
    filtered_df[[
        "track_name", "artist_name", "album_name", "release_year",
        "valence", "vader_compound", "sentiment_divergence", "melancholy_index"
    ]].sort_values(by="sentiment_divergence", ascending=False),
    width="stretch"
)
