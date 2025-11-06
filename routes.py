from flask import Blueprint
from controllers.home import home
from controllers.translator import (
    translate_char,
    translate_text
)

# Blueprints
home_bp = Blueprint('home', __name__)
translate_bp = Blueprint('translate', __name__)

# Routes
@home_bp.route('/', methods=['GET'])
def index():
    return home()

@translate_bp.route('/char', methods=['GET'])
def translate_one():
    return translate_char()

@translate_bp.route('/text', methods=['GET'])
def translate():
    return translate_text()

def register_routes(app):
    app.register_blueprint(home_bp, url_prefix='/')
    app.register_blueprint(translate_bp, url_prefix='/translate')
