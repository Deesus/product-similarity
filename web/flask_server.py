from flask.cli import FlaskGroup
from flask import Flask
from flask_cors import CORS
from backend.routes import api

app = Flask(__name__)
app.register_blueprint(api)

# enables CORS for ALL routes:
# TODO: in production, we should only allow cross-origin requests from the domain where the front-end app is hosted;
# See <https://flask-cors.readthedocs.io/en/latest/> for more info.
CORS(app)

cli = FlaskGroup(app)


if __name__ == '__main__':
    cli()
