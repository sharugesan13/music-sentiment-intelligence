import os
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

print("Validating Genius API...")
assert GENIUS_ACCESS_TOKEN, "Missing Genius access token in .env"

# In lyricsgenius >= 3.15, pass access_token and set verbose on the instance
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15)
genius.verbose = False

song = genius.search_song("Cardigan", "Taylor Swift")
if song:
    print(f"Genius authenticated successfully! Sample track fetched: {song.title}")
    print("Environment verified.")
else:
    print("Connected, but sample track not returned.")
