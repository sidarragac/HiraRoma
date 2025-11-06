import os

from flask import current_app
from werkzeug.utils import secure_filename

def save_image(image):
    upload_folder = current_app.config['UPLOAD_FOLDER']
    
    filename = secure_filename(image.filename)
    img_path = os.path.join(upload_folder, filename)
    
    image.save(img_path)

    return img_path