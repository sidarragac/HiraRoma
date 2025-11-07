# HiraRoma

**Proyecto final de la asignatura Inteligencia Artificial (SI3003)** de la Universidad EAFIT. 
Desarrollado por: 
- Juan Carlos Citelly G.
- Santiago Idárraga C.
- Mateo Pineda A.

## Descripción  
HiraRoma es una aplicación desarrollada con el fin de convertir caracteres del silabario japonés **hiragana** a su equivalente en **romanización (romaji)**. El sistema está diseñado como parte del cierre del curso de IA, implementando técnicas vistas en clase para el procesamiento de lenguaje y mapeo de caracteres.

## Características principales  
- Conversión de texto en hiragana a romaji.
- Traducción de romaji a español.    
- Código en Python junto con componentes web (HTML) para la demostración del funcionamiento del modelo de ML.  
- Estructura modular que permite extender el sistema (por ejemplo: incluir katakana, kanji, reglas de pronunciación, etc).

## Tecnologías utilizadas  
- Python (versión mínima: 3.13.5)  
- Framework Flask para la interfaz de usuario.    
- Librerías especificadas en `requirements.txt`.

## Instalación  
1. Clonar el repositorio:  
    ```bash
        git clone https://github.com/sidarragac/HiraRoma.git
        cd HiraRoma
    ```
2. Crear y activar un entorno virtual de python:
    ```bash
        python -m venv venv  
        venv\Scripts\activate
    ```
3. Instalación de librerias:
    ```bash
        pip install -r requirements.txt
    ```
4. Crear archivo `.env` con las variables requeridas en `.env.example`[^1]

## ¿Cómo correr el código?
1. Entrar en la carpeta app: 
    ```bash
        cd app
    ```
2. Ejecutar el siguiente comando:
    ```bash
        python app.py
    ```
[^1]: Para poder ejecutar el proyecto hay que tener una cuenta en `Google Cloud Platform` con permisos de uso de `Cloud Translate`
