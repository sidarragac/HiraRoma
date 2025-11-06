from utils.image_predictor import ImagePredictor
from utils.image_processor import ImageProcessor
from utils.transliterator import Transliterator

from flask import current_app, render_template
from utils.image_saver import save_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def translate_char():
    return render_template('translation/translate_char.html')

def process_char(image):
    if not allowed_file(image.filename):
        return {'error': 'Tipo de archivo no permitido'}, 400

    img_path = save_image(image)

    predictor = ImagePredictor()
    transliterator = Transliterator()

    predicted_char = predictor.predict_image(img_path)
    char, romanized_char = transliterator.transliterate(predicted_char)

    viewdata = {}
    viewdata['predicted_char'] = char
    viewdata['romanized_char'] = romanized_char

    return render_template('translation/processed_char.html', viewdata=viewdata)

def translate_text():
    return render_template('translation/translate_text.html')
