import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv

#Get API Ids from .env
load_dotenv()

cid = os.getenv('client_id')
secret = os.getenv('client_secret')

#Apply API Ids for Spotipy
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

#Example data grab
steely_uri = 'spotify:artist:6P7H3ai06vU1sGvdpBwDmE'
results = sp.artist_albums(steely_uri, album_type='album')
albums = results['items']
while results['next']:
    results = sp.next(results)
    albums.extend(results['items'])

for album in albums:
    print(album['name'])