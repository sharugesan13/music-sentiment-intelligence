import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 1. Load Processed Dataset
df = pd.read_csv("music_sentiment_intelligence.csv")

print("=" * 80)
print("MUSIC & LYRIC SENTIMENT INTELLIGENCE: STATISTICAL EXPLORATORY ANALYSIS")
print("=" * 80)

swift = df[df["artist_name"] == "Taylor Swift"]
lana = df[df["artist_name"] == "Lana Del Rey"]

# 2. Descriptive Summary Metrics
metrics = ["valence", "energy", "acousticness", "vader_compound", "sentiment_divergence", "melancholy_index"]

summary_table = df.groupby("artist_name")[metrics].agg(["mean", "std", "median"]).T
print("\n--- COMPARATIVE DESCRIPTIVE METRICS ---")
print(summary_table.round(3))

# 3. Two-Sample Kolmogorov-Smirnov Tests
print("\n--- TWO-SAMPLE KOLMOGOROV-SMIRNOV TESTS (Swift vs. Lana) ---")
print(f"{'Feature':<25} | {'KS Statistic':<15} | {'p-value':<12} | {'Significant (α=0.05)'}")
print("-" * 72)

for m in metrics:
    ks_stat, p_val = stats.ks_2samp(swift[m].dropna(), lana[m].dropna())
    sig = "YES (p < 0.05)" if p_val < 0.05 else "NO"
    print(f"{m:<25} | {ks_stat:<15.4f} | {p_val:<12.4e} | {sig}")

# 4. Identify High-Divergence Tracks ("The Trojan Horse / Deceptive Euphoria")
# Tracks with high musical brightness (Valence > 0.5) but negative lyrical sentiment (VADER < 0)
trojan_tracks = df[(df["valence"] >= 0.50) & (df["vader_compound"] < 0.0)][
    ["track_name", "artist_name", "album_name", "valence", "vader_compound", "sentiment_divergence"]
].sort_values(by="sentiment_divergence", ascending=False)

print("\n--- TOP DECEPTIVE EUPHORIA TRACKS (High Valence, Negative Lyrics) ---")
print(trojan_tracks.to_string(index=False))

# 5. Visualizations
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(16, 12))

# Subplot A: Melancholy Index & Valence Density Distributions (KDE)
ax1 = plt.subplot(2, 2, 1)
sns.kdeplot(data=df, x="melancholy_index", hue="artist_name", fill=True, common_norm=False, palette={"Taylor Swift": "#e056fd", "Lana Del Rey": "#0984e3"}, ax=ax1, alpha=0.4)
ax1.set_title("Distribution of Melancholy Index", fontsize=13, fontweight="bold")
ax1.set_xlabel("Melancholy Index (0.0 = Bright, 1.0 = Melancholic)")

ax2 = plt.subplot(2, 2, 2)
sns.kdeplot(data=df, x="valence", hue="artist_name", fill=True, common_norm=False, palette={"Taylor Swift": "#e056fd", "Lana Del Rey": "#0984e3"}, ax=ax2, alpha=0.4)
ax2.set_title("Distribution of Acoustic Valence", fontsize=13, fontweight="bold")
ax2.set_xlabel("Spotify Valence (0.0 = Gloomy, 1.0 = Euphoric)")

# Subplot B: Quadrant Scatter Plot (Lyrical Sentiment vs. Musical Brightness)
ax3 = plt.subplot(2, 1, 2)

palette = {"Taylor Swift": "#e056fd", "Lana Del Rey": "#0984e3"}
scatter = sns.scatterplot(
    data=df,
    x="vader_compound",
    y="valence",
    hue="artist_name",
    style="artist_name",
    s=120,
    alpha=0.85,
    palette=palette,
    ax=ax3
)

# Reference dividing thresholds
ax3.axvline(0.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax3.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)

# Quadrant contextual labels
ax3.text(-0.95, 0.93, "QUADRANT II: DECEPTIVE EUPHORIA\n(Sad Lyrics, Upbeat Beat)", fontsize=10, fontweight="bold", color="#d63031", bbox=dict(facecolor="white", alpha=0.7, edgecolor="#d63031"))
ax3.text(0.40, 0.93, "QUADRANT I: UNIFIED OPTIMISM\n(Happy Lyrics, Upbeat Beat)", fontsize=10, fontweight="bold", color="#27ae60", bbox=dict(facecolor="white", alpha=0.7, edgecolor="#27ae60"))
ax3.text(-0.95, 0.08, "QUADRANT III: PURE MELANCHOLIA\n(Sad Lyrics, Somber Sound)", fontsize=10, fontweight="bold", color="#2c3e50", bbox=dict(facecolor="white", alpha=0.7, edgecolor="#2c3e50"))
ax3.text(0.40, 0.08, "QUADRANT IV: AMBIENT REFLECTION\n(Happy Lyrics, Somber Sound)", fontsize=10, fontweight="bold", color="#e67e22", bbox=dict(facecolor="white", alpha=0.7, edgecolor="#e67e22"))

# Annotate sample outlier tracks
sample_tracks = df.sort_values(by="sentiment_divergence", ascending=False).head(3).to_dict("records") + \
                df.sort_values(by="melancholy_index", ascending=False).head(2).to_dict("records")

annotated = set()
for item in sample_tracks:
    if item["track_name"] not in annotated:
        ax3.annotate(
            item["track_name"],
            (item["vader_compound"], item["valence"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            fontweight="semibold",
            alpha=0.9
        )
        annotated.add(item["track_name"])

ax3.set_xlim(-1.05, 1.05)
ax3.set_ylim(-0.05, 1.05)
ax3.set_title("Lyric Sentiment (VADER) vs. Sonic Valence (Spotify)", fontsize=14, fontweight="bold", pad=12)
ax3.set_xlabel("Normalized Lyrical Sentiment (Compound VADER: -1.0 to +1.0)", fontsize=11)
ax3.set_ylabel("Musical Valence (Acoustic Brightness: 0.0 to 1.0)", fontsize=11)
ax3.legend(title="Artist", loc="lower left", frameon=True)

plt.tight_layout()
plt.savefig("eda_sentiment_quadrants.png", dpi=300)
print("\nSaved high-resolution chart to: eda_sentiment_quadrants.png")
