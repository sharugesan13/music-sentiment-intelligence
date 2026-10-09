import os
import re
import pandas as pd
from dotenv import load_dotenv
import lyricsgenius
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

load_dotenv()
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Correct lyricsgenius >= 3.15 initialization
genius = lyricsgenius.Genius(
    GENIUS_TOKEN, 
    timeout=10, 
    remove_section_headers=True
)
genius.verbose = False

sia = SentimentIntensityAnalyzer()
audio_df = pd.read_csv("filtered_audio_features.csv")

def clean_lyrics(text):
    if not text:
        return ""
    # Strip headers, brackets, metadata and footer tokens
    text = re.sub(r'^\d+\s*Contributors.*Lyrics', '', text)
    text = re.sub(r'\d*Embed$', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

print(f"Processing NLP sentiment for {len(audio_df)} tracks...")

records = []
for idx, row in audio_df.iterrows():
    track = row["track_name"]
    artist = row["artist_name"]
    
    clean_title = re.sub(r"\(.*?\)", "", track).strip()
    
    try:
        song = genius.search_song(clean_title, artist)
        raw_lyrics = song.lyrics if song else ""
    except Exception as e:
        raw_lyrics = ""
    
    lyrics = clean_lyrics(raw_lyrics)
    
    # NLP metrics
    if lyrics:
        vader_scores = sia.polarity_scores(lyrics)
        vader_compound = vader_scores["compound"]
        blob = TextBlob(lyrics)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        words = re.findall(r'\b\w+\b', lyrics.lower())
        lexical_density = len(set(words)) / len(words) if words else 0.0
    else:
        vader_compound = 0.0
        polarity = 0.0
        subjectivity = 0.0
        lexical_density = 0.0
    
    valence = row["valence"]
    energy = row["energy"]
    
    # 1. Sentiment Divergence (Trojan Horse index: high = cheerful beat + sad lyric)
    norm_vader = (vader_compound + 1.0) / 2.0
    sentiment_divergence = valence - norm_vader
    
    # 2. Composite Melancholy Index: [0.0 = bright/upbeat, 1.0 = deeply melancholic]
    melancholy_index = ((1.0 - valence) * 0.4) + ((1.0 - energy) * 0.3) + (((1.0 - vader_compound) / 2.0) * 0.3)
    
    rec = row.to_dict()
    rec["lyrics_found"] = bool(lyrics)
    rec["vader_compound"] = round(vader_compound, 4)
    rec["textblob_polarity"] = round(polarity, 4)
    rec["textblob_subjectivity"] = round(subjectivity, 4)
    rec["lexical_density"] = round(lexical_density, 4)
    rec["sentiment_divergence"] = round(sentiment_divergence, 4)
    rec["melancholy_index"] = round(melancholy_index, 4)
    
    records.append(rec)
    print(f"[{idx+1}/{len(audio_df)}] {track} ({artist}) | VADER: {round(vader_compound, 2)} | Valence: {round(valence, 2)} | Divergence: {round(sentiment_divergence, 2)}")

final_df = pd.DataFrame(records)
final_df.to_csv("music_sentiment_intelligence.csv", index=False)
print("\nExtraction & NLP Pipeline complete! Saved to: music_sentiment_intelligence.csv")
