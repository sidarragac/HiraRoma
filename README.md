# Hira2Roma
Proyecto final correspondiente a la materia Inteligencia Artificial (SI3003) de la Universidad EAFIT

## Arquitectura del Proyecto

Este proyecto utiliza Flask como framework web con una arquitectura modular que incluye:

- **Controladores**: Manejan las peticiones HTTP
- **Utils**: Utilidades para procesamiento de imágenes y conexión a Google APIs
- **Config**: Configuración de la aplicación

### Estructura del Proyecto

```
Hira2Roma/
├── app.py                          # Punto de entrada de la aplicación
├── config/
│   ├── __init__.py
│   └── config.py                   # Configuración de Flask
├── controllers/
│   ├── __init__.py
│   ├── image_controller.py         # Controlador de imágenes
│   └── google_api_controller.py    # Controlador de Google APIs
├── utils/
│   ├── __init__.py
│   ├── image_processor.py          # Procesamiento de imágenes
│   └── google_api_client.py        # Cliente de Google APIs
├── uploads/                        # Carpeta para imágenes subidas
├── processed/                      # Carpeta para imágenes procesadas
├── requirements.txt                # Dependencias del proyecto
├── .env.example                    # Ejemplo de variables de entorno
├── .gitignore
└── README.md
```

## Instalación

1. **Clonar el repositorio**
```bash
git clone <url-del-repositorio>
cd Hira2Roma
```

2. **Crear entorno virtual**
```bash
python -m venv venv
```

3. **Activar entorno virtual**
- Windows:
```bash
venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

4. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

5. **Configurar variables de entorno**
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

6. **Configurar Google Cloud**
- Crea un proyecto en [Google Cloud Console](https://console.cloud.google.com/)
- Habilita las APIs: Vision API y Translation API
- Descarga las credenciales JSON
- Configura la variable `GOOGLE_APPLICATION_CREDENTIALS` en `.env`

## Uso

### Ejecutar la aplicación

```bash
python app.py
```

La aplicación estará disponible en `http://localhost:5000`

### Endpoints Disponibles

#### Imágenes (`/api/images`)

- **POST /api/images/upload** - Subir una imagen
- **POST /api/images/process** - Procesar una imagen
- **POST /api/images/convert** - Convertir formato de imagen
- **GET /api/images/download/<filename>** - Descargar imagen procesada
- **GET /api/images/info/<filename>** - Obtener información de imagen

#### Google APIs (`/api/google`)

- **POST /api/google/vision/analyze** - Análisis completo de imagen
- **POST /api/google/vision/detect-text** - Detectar texto en imagen
- **POST /api/google/vision/detect-labels** - Detectar etiquetas
- **POST /api/google/translate** - Traducir texto
- **GET /api/google/health** - Verificar estado de conexión

## Ejemplos de Uso

### Subir y procesar una imagen

```bash
# Subir imagen
curl -X POST -F "file=@imagen.jpg" http://localhost:5000/api/images/upload

# Procesar imagen (redimensionar y convertir a escala de grises)
curl -X POST http://localhost:5000/api/images/process \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "imagen.jpg",
    "operations": ["resize", "grayscale"],
    "params": {"width": 800}
  }'
```

### Analizar imagen con Google Vision

```bash
curl -X POST http://localhost:5000/api/google/vision/analyze \
  -H "Content-Type: application/json" \
  -d '{"filename": "imagen.jpg"}'
```

### Traducir texto

```bash
curl -X POST http://localhost:5000/api/google/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "target_language": "es"
  }'
```

## Operaciones de Procesamiento de Imágenes

El `ImageProcessor` soporta las siguientes operaciones:

- `resize` - Redimensionar imagen
- `crop` - Recortar imagen
- `rotate` - Rotar imagen
- `flip` - Voltear imagen (horizontal/vertical)
- `grayscale` - Convertir a escala de grises
- `brightness` - Ajustar brillo
- `contrast` - Ajustar contraste
- `sharpness` - Ajustar nitidez
- `blur` - Aplicar desenfoque
- `edge_detection` - Detección de bordes
- `emboss` - Efecto de relieve
- `invert` - Invertir colores

## Tecnologías Utilizadas

- **Flask**: Framework web
- **Pillow**: Procesamiento de imágenes
- **Google Cloud Vision API**: Análisis de imágenes
- **Google Cloud Translation API**: Traducción de texto
- **python-dotenv**: Gestión de variables de entorno

## Licencia

Este proyecto es parte de un proyecto académico de la Universidad EAFIT.
