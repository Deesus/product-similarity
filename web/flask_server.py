from flask.cli import FlaskGroup
from flask import Flask
from backend.routes import api

app = Flask(__name__)
app.register_blueprint(api)

cli = FlaskGroup(app)


if __name__ == '__main__':
    cli()
