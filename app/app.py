from flask import Flask, Blueprint, send_from_directory
from config.config import Config
from routes import register_routes

def create_app(config_class=Config):
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    app.config.from_object(config_class)
    
    register_routes(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=8000)
