import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from DataStorage import DataStorage 

# Get API Ids from .env
load_dotenv()

cid = os.getenv('client_id')
secret = os.getenv('client_secret')

# Apply API Ids for Spotipy
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_tracks_by_genre(genre, limit=500):
    # Ensure limit is an integer
    limit = int(limit)  # Convert the limit to an integer

    track_data = []
    search_limit = 50  # Maximum number of tracks returned by the Spotify API in a single request
    offset = 0

    while len(track_data) < limit:
        results = sp.search(q=f'genre:{genre}', type='track', limit=search_limit, offset=offset)
        tracks = results['tracks']['items']
        
        if not tracks:  
            break

        track_data.extend(tracks)
        offset += search_limit
        time.sleep(1)  # To avoid hitting rate limits

    return track_data[:limit]

def get_audio_features(track_ids):
    # Fetches the audio features for a list of track IDs.
    audio_features = []
    batch_size = 100  # Spotify allows up to 100 track IDs per request
    
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        audio_features_batch = sp.audio_features(batch)
        audio_features.extend(audio_features_batch)
    
    return audio_features

def display_track_info(tracks, audio_features):
    # Displays the track name, popularity, and audio features for each track.
    for idx, track in enumerate(tracks):
        track_name = track['name']
        track_duration = track["duration_ms"],
        track_popularity = track['popularity']
        features = audio_features[idx]

        if features:
            print(f"Track {idx+1}:")
            print(f"  Name: {track_name}")
            print(f"  Duration (ms): {track_duration}")
            print(f"  Popularity: {track_popularity}")
            print(f"  Danceability: {features['danceability']}")
            print(f"  Energy: {features['energy']}")
            print(f"  Loudness: {features['loudness']}")
            print(f"  Tempo: {features['tempo']}")
            print(f"  Acousticness: {features['acousticness']}")
            print(f"  Speechiness: {features['speechiness']}")
            print(f"  Instrumentalness: {features['instrumentalness']}")
            print(f"  Liveness: {features['liveness']}")
            print(f"  Valence: {features['valence']}")
            print()

def save_tracks_to_csv(tracks, audio_features, filename='spotify_tracks.csv'):
    # Combine track and audio feature data for saving to CSV
    track_data = []
    for idx, track in enumerate(tracks):
        features = audio_features[idx]
        if features:
            track_data.append({
                'Track_Name': track['name'],
                'Track Duration (ms)': track["duration_ms"],
                'Popularity': track['popularity'],
                'Danceability': features['danceability'],
                'Energy': features['energy'],
                'Loudness': features['loudness'],
                'Tempo': features['tempo'],
                'Acousticness': features['acousticness'],
                'Speechiness': features['speechiness'],
                'Instrumentalness': features['instrumentalness'],
                'Liveness': features['liveness'],
                'Valence': features['valence']
            })

    headers = ['Track_Name', 'Track Duration (ms)', 'Popularity', 'Danceability', 'Energy', 'Loudness', 'Tempo', 'Acousticness', 
               'Speechiness', 'Instrumentalness', 'Liveness', 'Valence']
    
    # Use DataStorage to save the data
    storage = DataStorage(filename)
    storage.save_to_csv(track_data, headers)
    print(f"Track data has been saved to {filename}.")

# Main Execution
if __name__ == '__main__':
    # Example of genre input (replace with the genre you want)
    genre_input = input("Enter the genre you want to search for: ")
    results_input = input("Enter the number of desired results: ")

    # Fetch tracks for the specified genre
    tracks = get_tracks_by_genre(genre_input, results_input)

    # Extract track IDs for fetching audio features
    track_ids = [track['id'] for track in tracks]

    # Fetch audio features for all tracks
    audio_features = get_audio_features(track_ids)

    # Display the track name, popularity, and audio features
    display_track_info(tracks, audio_features)

    # Save the track and audio feature data to a CSV file
    save_tracks_to_csv(tracks, audio_features, genre_input + "_" + results_input + ".csv")
