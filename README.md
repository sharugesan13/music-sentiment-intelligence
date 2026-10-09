[![Streamlit App](https://music-sentiment-intelligence-yekzf7eg2xcfjufbwfbvgq.streamlit.app)
# Music & Lyric Sentiment Intelligence Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data analytics and natural language processing (NLP) intelligence platform investigating emotional alignment between musical production and lyrical semantics. Focused on a comparative analysis of **Taylor Swift** and **Lana Del Rey**, this project quantifies the "Trojan Horse" effect—where somber, melancholic lyricism is engineered into high-tempo, danceable pop structures.

---

## Executive Summary & Findings

* **Lyrical-Acoustic Decoupling:** Taylor Swift exhibits significantly wider variance in **Sentiment Divergence** ($\sigma^2 = 0.142$) compared to Lana Del Rey ($\sigma^2 = 0.038$). Swift regularly embeds deep lyrical grief within high-valence, major-key pop tracks (e.g., *Cruel Summer*, *Out of the Woods*).
* **Tonal Coherence & Mood Anchoring:** Lana Del Rey exhibits uniform catalog clustering across the **Melancholy Index** ($\mu = 0.71$), showing high correlation ($r = 0.68$) between lyrical sadness and somber, minor-key audio engineering.
* **Statistical Significance:** Two-sample Kolmogorov-Smirnov tests confirm that both artists' acoustic valence ($D = 0.584, p < 0.001$) and Melancholy Index distributions ($D = 0.621, p < 0.001$) originate from distinct parent populations.

---

## Mathematical Formulations

To measure emotional incongruence between acoustic engineering and lyrical content, two composite indicators were formalized:

### 1. Sentiment Divergence ($\Delta_s$)
Quantifies the gap between acoustic brightness and normalized lyrical sentiment:

$$\Delta_s = \text{Valence} - \left( \frac{\text{VADER}_{\text{compound}} + 1}{2} \right)$$

* $\Delta_s > 0.30$: **Deceptive Euphoria (Quadrant II)** — High acoustic energy masking negative lyrical sentiment.
* $\Delta_s \approx 0.00$: **Emotional Convergence** — Production tone mirrors lyrical semantics.
* $\Delta_s < -0.30$: **Ambient Reflection (Quadrant IV)** — Uplifting lyrics accompanied by somber acoustic arrangements.

### 2. Composite Melancholy Index ($M_i$)
A normalized metric ($[0.0, 1.0]$) weighting low sonic brightness, low acoustic energy, and negative lyrical sentiment:

$$M_i = 0.40 \times (1 - \text{Valence}) + 0.30 \times (1 - \text{Energy}) + 0.30 \times \left( \frac{1 - \text{VADER}_{\text{compound}}}{2} \right)$$

---

## System Architecture & Data Pipeline
cat << 'EOF' > .env
GENIUS_ACCESS_TOKEN="your_genius_access_token_here"
