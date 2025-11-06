from flask import Blueprint, request
from controllers.home import home
from controllers.translator import (
    translate_text,
    process_text
)

# Blueprints
home_bp = Blueprint('home', __name__)
translate_bp = Blueprint('translate', __name__)

# Routes
@home_bp.route('/', methods=['GET'])
def index():
    return home()

@translate_bp.route('/', methods=['GET'])
def translate():
    return translate_text()

@translate_bp.route('/result', methods=['POST'])
def translate_result():
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    image = request.files['image']
    return process_text(image)

def register_routes(app):
    app.register_blueprint(home_bp, url_prefix='/')
    app.register_blueprint(translate_bp, url_prefix='/translate')
