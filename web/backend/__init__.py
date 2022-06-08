from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# enables CORS for ALL routes:
# TODO: in production, we should only allow cross-origin requests from the domain where the front-end app is hosted;
# See <https://flask-cors.readthedocs.io/en/latest/> for more info.
CORS(app)
