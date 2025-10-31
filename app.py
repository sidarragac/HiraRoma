"""
Aplicación principal Flask para Hira2Roma
"""
from flask import Flask
from config.config import Config
from controllers.image_controller import image_bp

def create_app(config_class=Config):
    """Factory para crear la aplicación Flask"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Registrar blueprints (controladores)
    app.register_blueprint(image_bp, url_prefix='/api/images')
    
    # Ruta de prueba
    @app.route('/')
    def index():
        return {
            'message': 'Hira2Roma API',
            'version': '1.0.0',
            'endpoints': {
                'images': '/api/images',
                'google': '/api/google'
            }
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='127.0.0.1', port=5000)
