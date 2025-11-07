from flask import Blueprint, request, send_from_directory
from config.config import Config
from controllers.translator import (
    translate_text,
    process_text
)

# Blueprints
translate_bp = Blueprint('translate', __name__)
uploads_bp = Blueprint('uploads', __name__, url_prefix='/uploads')

# Routes
@translate_bp.route('/', methods=['GET', 'POST'])
def translate_result():
    if request.method == 'GET':
        return translate_text()
    
    if 'image' not in request.files:
        return {'error': 'No image provided'}, 400
    image = request.files['image']
    return process_text(image)

@uploads_bp.route('/<path:filename>')
def uploaded_file(filename):
    if filename.startswith('uploads/'):
        filename = filename[len('uploads/'):]

    return send_from_directory(Config.UPLOAD_FOLDER, filename)

def register_routes(app):
    app.register_blueprint(translate_bp, url_prefix='/')
    app.register_blueprint(uploads_bp)
