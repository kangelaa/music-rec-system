import os
from flask import Flask

#create an instance of a Flask app called 'app' and indicate its location
app_path = os.path.dirname(__file__)
app = Flask(__name__, static_folder=app_path+'/static')
#create a secret app config key for security
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba241'

from application import routes
