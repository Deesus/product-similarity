from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
# enables CORS for ALL routes:
CORS(app)
