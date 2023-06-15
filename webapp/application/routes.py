from application import app
from flask import render_template, request
from application.features import *
from application.model import *

#load in pre-generated song column data, drop nas and reset index in place (CHANGE FILE PATH TO YOUR LOCAL COMPUTERS FILE STRUCTURE)
songDF = pd.read_csv("./data/allsong_columns.csv",index_col=0) #pre-engineering features
songDF.dropna(inplace=True)

#currently loading in subset of data (NOTE: running full_features from the .zip will require patience and a fast computer!)
# load in feature set - (host website/Github requires us to batch this, as max file size is 100MB)
feat_set1 = pd.read_csv("./data/features1.csv",index_col=0) #post-engineering features (IF USING FULL_FEATURES, CHANGE FILE PATH TO YOUR LOCAL COMPUTERS FILE STRUCTURE)
feat_set2 = pd.read_csv("./data/features2.csv",index_col=0)
complete_feature_set = pd.concat([feat_set1,feat_set2],axis=0)
complete_feature_set.dropna(inplace=True)
complete_feature_set.reset_index(inplace=True,drop=True)

# routes for each webapp page
@app.route("/")
def home():
   #render the home page
   return render_template('home.html')

@app.route("/about")
def about():
   #render the about page
   return render_template('about.html')

@app.route('/recommend', methods=['POST'])
def recommend():
   #request the URL from the HTML form
   URL = request.form['URL']
   #use the extract function to get a feature set dataframe from the input playlist URL
   df = extract(URL)
   #retrieve the results and get as many recommendations as the user requested
   finalrecs = recommend_from_playlist(songDF, complete_feature_set, df)
   number_of_recs = int(request.form['number-of-recs'])
   my_songs = []
   for i in range(number_of_recs):
      #append info to display on website for each song to my_songs
      my_songs.append([str(finalrecs.iloc[i,3]) + ' - '+ '"'+str(finalrecs.iloc[i,2])+'"', "https://open.spotify.com/track/"+ str(finalrecs.iloc[i,-6]).split("/")[-1]])
   return render_template('results.html',songs= my_songs)