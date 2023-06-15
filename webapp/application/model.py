from sklearn.metrics.pairwise import cosine_similarity
from application.features import *

def generate_playlist_feature(complete_feature_set, playlist_df):
    '''
    Summarize a user's playlist into a single vector obtain non-playlist song database feature set
    ---
    Input: 
    complete_feature_set (pandas dataframe): Dataframe which includes all of the processed features for the database of songs
    playlist_df (pandas dataframe): playlist dataframe of unprocessed playlist feature columns
        
    Output: 
    complete_feature_set_playlist_final (pandas series): single vector feature that summarizes the playlist
    complete_feature_set_nonplaylist (pandas dataframe): feature set for songs not in the playlist
    '''
    
    # generate processed feature sets from input playlist columns created from extract()
    complete_feature_set_playlist, playlist_genres = create_feature_set(playlist_df)
    complete_feature_set = complete_feature_set

    # generate genre trimmed feature sets from genres that the two feature sets share (for cosine similarity comparison)
    commoncols = complete_feature_set.columns.intersection(complete_feature_set_playlist.columns)
    genretrimmed_feature_set = complete_feature_set[commoncols]
    genretrimmed_playlist_set = complete_feature_set_playlist[commoncols]

    # Find all non-playlist song features
    complete_feature_set_nonplaylist = genretrimmed_feature_set[~genretrimmed_feature_set['id'].isin(playlist_df['id'].values)]

    #drop id col from playlist feature set
    complete_feature_set_playlist_final = genretrimmed_playlist_set.drop(columns = "id")

    #return both feature sets
    return complete_feature_set_playlist_final.sum(axis = 0), complete_feature_set_nonplaylist

def generate_playlist_recos(df, features, nonplaylist_features):
    '''
    Generated recommendation based on cosine distance between summarized songs in a specific playlist and non-playlist feature set database songs
    ---
    Input: 
    df (pandas dataframe): spotify dataframe with all unprocessed song data
    features (pandas series): summarized playlist features (single vector)
    nonplaylist_features (pandas dataframe): feature set of songs that are not in the selected playlist
        
    Output: 
    non_playlist_df_top_40: Top 40 song recommendations for that playlist based on cosine similarity
    '''

    #make df of songs not in playlist
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    #drop nas in playlist feature vector
    features.dropna(inplace=True)
    # Find cosine similarity between the playlist and the complete song set
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    #get top 40 most similar songs
    non_playlist_df_top_40 = non_playlist_df.sort_values('sim',ascending = False).head(40)
    
    return non_playlist_df_top_40

def recommend_from_playlist(songDF,complete_feature_set,playlistDF_test):
    '''
    Run through pipeline to generate playlist feature set, get non-playlist feature set, and identify top 40 song recommendations using the 2 functions above
    '''
    # Find feature sets
    complete_feature_set_playlist_vector, complete_feature_set_nonplaylist = generate_playlist_feature(complete_feature_set, playlistDF_test)
    
    # Generate recommendations
    top40 = generate_playlist_recos(songDF, complete_feature_set_playlist_vector, complete_feature_set_nonplaylist)

    return top40