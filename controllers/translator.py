from flask import current_app, render_template

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def translate_char():
    return render_template('translate_char.html')

def translate_text():
    return render_template('translate_text.html')