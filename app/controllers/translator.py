import os

from utils.image_predictor import ImagePredictor
from utils.image_processor import ImageProcessor
from utils.transliterator import Transliterator

from flask import current_app, render_template
from utils.image_saver import save_image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def translate_text():
    viewData = {}
    viewData['count'] = 0
    return render_template('translation/translate.html', viewData=viewData)

def process_text(image):
    if not allowed_file(image.filename):
        return {'error': 'Tipo de archivo no permitido'}, 400

    img_path = save_image(image)
    folder_path = os.path.dirname(img_path)

    processor = ImageProcessor(img_path, 500)
    predictor = ImagePredictor()
    transliterator = Transliterator()

    count = processor.get_grid(folder_path)


    viewData = {}
    viewData['count'] = count

    if count == 1:
        predicted_char = predictor.predict_image(img_path)
        char, romanized_char = transliterator.transliterate(predicted_char)
        
        image_files = [f"{folder_path}/{f}" for f in os.listdir(folder_path)]
        viewData['zip'] = [(char, romanized_char, image_files[0])]

        viewData['hiragana_word'] = ''
        viewData['romanized_word'] = ''
        return render_template('translation/translate.html', viewData=viewData)

    predicted_char =  predictor.predict_images_in_folder(folder_path)
    res = transliterator.transliterate_text(predicted_char)    
    
    predicted_arr = []
    romanized_arr = []
    for char, romanized_char in res:
        predicted_arr.append(char)
        romanized_arr.append(romanized_char)

    image_files = [f"{folder_path}/{f}" for f in os.listdir(folder_path)]
    viewData['zip'] = list(zip(predicted_arr, romanized_arr, image_files))

    viewData['hiragana_word'] = ''.join(predicted_arr)
    viewData['romanized_word'] = ''.join(romanized_arr)

    return render_template('translation/translate.html', viewData=viewData)
