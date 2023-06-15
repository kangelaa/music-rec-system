import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import time

def drop_duplicates_byname(df):
    '''
    Drop song duplicates

    Input: full dataframe of songs with possible duplicates
    Output: dataframe of songs with duplicates dropped
    '''
    df['songartistconcat'] = df.apply(lambda row: row['artist_name']+row['track_name'],axis = 1)
    return df.drop_duplicates('songartistconcat')

def extract(URL):
    '''
    Get features from user input playlist

    :param playlist URL:
    :return: playlist complete feature set
    '''
    #desiree's client
    client_id = "79d3fd66880741beb44570cd46351fde"
    client_secret = "19464e980ae34fff8faea2bfd09d7851"

    #use the clint secret and id details
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # the URI is split by ':' to get the username and playlist ID
    playlist_id = URL.split("/")[4].split("?")[0]
    tracks = sp.playlist_tracks(playlist_id)

    #lists that will be filled in with features
    track_id = []
    playlist_audio_features = []
    playlist_misc_info = []
    playlist_artist_genres = []
    playlist_artist_popularity = []
    playlist_artist_ids = []
    playlist_artist_name = []
    playlist_album_name=[]

    #loop through tracks and extract id
    for track in tracks['items']: #change this for if you're having user sign in
        track_id.append(track['track']['id'])

    #get groups of 100 chunks
    hundred_uri_chunks = [track_id[i:i + 100] for i in range(0, len(track_id), 100)]

    #get audio features
    for chunk in hundred_uri_chunks:
        chunk_features = sp.audio_features(chunk)
        playlist_audio_features.extend(chunk_features)

    # get 50 track_ids at a time for sp.tracks()
    fifty_uri_chunks = [track_id[i:i + 50] for i in range(0, len(track_id), 50)]

    #get track name/popularity and artist ids
    for i, chunk in enumerate(fifty_uri_chunks):
        chunk_tracks = sp.tracks(chunk)['tracks']
        playlist_misc_info.extend(chunk_tracks)
        # use time.sleep() to avoid surpassing rate limit from API for every 1000 chunks
        if i+1 % 1000 == 0:
            time.sleep(30)

    #filter out None values
    playlist_misc_info = [item for item in playlist_misc_info if item is not None]
    playlist_misc_info = pd.DataFrame(playlist_misc_info)

    #save track popularity, artists, and track name
    playlist_track_popularity = playlist_misc_info["popularity"]
    playlist_track_name = playlist_misc_info["name"]
    playlist_artists = playlist_misc_info["artists"]
    playlist_albums = playlist_misc_info["album"]

    for album in playlist_albums:
        playlist_album_name.append(album["name"])

    #get total list of artist ids
    for artist in playlist_artists:
        # take first/main artist for each song
        playlist_artist_ids.append(artist[0]["id"])

    #get 50 artist_ids at a time for sp.artists()
    fifty_artist_uri_chunks = [playlist_artist_ids[i:i + 50] for i in range(0, len(playlist_artist_ids), 50)]

    #iterate through chunks to get artist popularity, name, and genre
    for i, chunk in enumerate(fifty_artist_uri_chunks):
        playlist_artist_info = sp.artists(chunk)
        playlist_artist_info = pd.DataFrame(playlist_artist_info["artists"])
        playlist_artist_popularity.extend(playlist_artist_info["popularity"])
        playlist_artist_genres.extend(playlist_artist_info["genres"])
        playlist_artist_name.extend(playlist_artist_info['name'])
        if i+1 % 1000 == 0:
            time.sleep(30)

    #concatenate all columns
    playlist_df = pd.concat([playlist_track_name, playlist_track_popularity, pd.Series(playlist_album_name),
                             pd.Series(playlist_artist_popularity), pd.Series(playlist_artist_genres),
                             pd.DataFrame(playlist_audio_features), pd.Series(playlist_artist_name)], axis=1)

    #rename playlist columns
    playlist_df = playlist_df.rename(
        columns = {0: 'album_name', 1: 'artist_popularity', "popularity": "track_popularity", 3: "artist_name",
                  2: "genres", "name": "track_name"})

    #drop duplicates and reset index
    playlist_df = drop_duplicates_byname(playlist_df)
    playlist_df.reset_index(drop=True, inplace=True)
    #drop na's and reset index
    playlist_df_final = playlist_df.dropna()
    playlist_df_final.reset_index(drop=True, inplace=True)

    return playlist_df_final

def ohe_prep(df, column, new_name):
    '''
    Create one-hot-encoded features of a specific column

    Input:
    df (pandas dataframe): Spotify Dataframe
    column (str): Column to be processed
    new_name (str): new column name to be used

    Output:
    tf_df: One-hot encoded features
    '''

    # get_dummies() converts categorical variable into dummy/indicator variables
    tf_df = pd.get_dummies(df[column])
    print(tf_df)
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop=True, inplace=True)
    return tf_df

def getSubjectivity(text):
  '''
  Getting the Text Subjectivity using TextBlob
  '''
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
  '''
  Getting the Text Polarity using TextBlob
  '''
  return TextBlob(text).sentiment.polarity

def getAnalysis(score, task="polarity"):
  '''
  Categorizing the Polarity & Subjectivity score (3 categories)
  '''
  if task == "subjectivity":
    if score < 1/3:
      return "low"
    elif score > 1/3:
      return "high"
    else:
      return "medium"
  else:
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'

def sentiment_analysis(df, text_col):
  '''
  Perform sentiment analysis on text
  ---
  Input:
  df (pandas dataframe): Dataframe of interest
  text_col (str): column of interest
  '''
  df['subjectivity'] = df[text_col].apply(getSubjectivity).apply(lambda x: getAnalysis(x,"subjectivity"))
  df['polarity'] = df[text_col].apply(getPolarity).apply(getAnalysis)
  return df

def create_feature_set(df):
    '''
    Process input playlist from playlist columns df (extract() output) to create a final set of normalized features that will be used to compare cosine similarity
    ---
    Input:
    df (pandas dataframe): input playlist Dataframe

    Output:
    final (pandas dataframe): Final set of processed input playlist features
    fullgenrescols (list): Final set of genres columns
    '''

    # TF-IDF implementation: find most important genre for each song and that genre's prevalence across all songs to weight genre accordingly
    # function from scikit-learn
    tfidf = TfidfVectorizer()
    # get weighted values for each genre
    df['genres_list'] = df['genres'].apply(lambda x: str(x).split(", "))
    tfidf_matrix = tfidf.fit_transform(df['genres_list'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names_out()]
    if 'genre|unknown' in genre_df.columns:
        genre_df.drop(columns='genre|unknown')  # Drop unknown genre
    fullgenrescols = list(genre_df.columns)
    # reset index col
    genre_df.reset_index(drop=True, inplace=True)

    # one-hot encoding
    key_ohe = ohe_prep(df, 'key', 'key') * 0.5  # keep data range in same range as other scaled numbers
    mode_ohe = ohe_prep(df, 'mode', 'mode') * 0.5
    time_signature = ohe_prep(df, 'time_signature', "time_signature") * 0.5

    # Sentiment analysis
    track_sentiment = sentiment_analysis(df, "track_name")
    album_sentiment = sentiment_analysis(df, "album_name")
    # ohe for sentiment analysis data
    track_subject_ohe = ohe_prep(track_sentiment, 'subjectivity',
                                 'tracksubjectivity') * 0.25  # weigh less because sentiment analysis less effective on short text
    track_polar_ohe = ohe_prep(track_sentiment, 'polarity', 'trackpolarity') * 0.25
    album_subject_ohe = ohe_prep(album_sentiment, 'subjectivity', 'albumsubjectivity') * 0.25
    album_polar_ohe = ohe_prep(album_sentiment, 'polarity', 'albumpolarity') * 0.25

    # scale audio columns
    audiofeature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
                         'liveness', 'valence', 'tempo']
    floats = df[audiofeature_cols].reset_index(drop=True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns=floats.columns)  # * 0.2

    # artist pop normalization
    artist_pop = df[["artist_popularity"]].reset_index(drop=True)
    scaler = MinMaxScaler()  # from scikit-learn
    artist_pop_scaled = pd.DataFrame(scaler.fit_transform(artist_pop), columns=artist_pop.columns)

    # track pop normalization
    track_pop = df[["track_popularity"]].reset_index(drop=True)
    scaler = MinMaxScaler()  # from scikit-learn
    track_pop_scaled = pd.DataFrame(scaler.fit_transform(track_pop), columns=track_pop.columns)

    # Concatenate all features
    final = pd.concat(
        [genre_df, key_ohe, mode_ohe, time_signature, track_subject_ohe, track_polar_ohe, album_subject_ohe,
         album_polar_ohe, floats_scaled, artist_pop_scaled, track_pop_scaled], axis=1)

    # Add song id
    final['id'] = df['id'].values

    return final, fullgenrescols
