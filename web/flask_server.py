from flask import Flask, render_template
from flask_cors import CORS
from backend.routes import api
import os


app = Flask(
    __name__,
    static_url_path='',
    static_folder='./dist',
    template_folder='./dist'
)
app.register_blueprint(api)

# In development, we need CORS since front-end and back-end are separated:
if os.environ.get('FLASK_ENV') == 'development':
    CORS(app)
# In production, we serve static assets from Flask, so we avoid CORS all together:
else:
    # Serve front-end:
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    # Pass-through 404 errors; the SPA/front-end will handle 404 pages:
    @app.errorhandler(404)
    def catch_all(path):
        return render_template('index.html')
