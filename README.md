# Spotify Recommendation System

Our project addresses the problem of finding new music: users have different preferences and tastes, and it can be difficult and time-consuming for them to find new, enjoyable songs to listen to. We approach this problem by building a content-based song recommendation system. 

To build our dataset of songs, we utilized a quarter of this dataset, the Spotify Million Playlist Dataset, to obtain tracks from thousands of different public playlists and use the track URI to extract additional track/artist features from the API for each song. Once we had a final preliminary dataset with all the features available in the Spotify API, we passed this dataset through a feature engineering pipeline that processed each feature column accordingly to translate all the songs into vectors, giving us a final feature set database. 

To analyze a user input playlist, we applied the same data extraction and feature engineering pipeline to the playlist. Then, we summarized the playlist by adding together each of the song vectors to obtain a final playlist vector to compare against each song in our database. Lastly, we built a Flask dynamic website and deployed it with PythonAnywhere to give users access to our recommendation system. The final product of our project is a website that takes in the input playlist of a user, and outputs a set of recommended songs based on the number of desired recommendations specified by the user. 

