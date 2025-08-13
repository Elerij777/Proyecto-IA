# Integrantes

Carlos David Trejo Mejía 20212000569 
Ángel Andrés Carias Castillo 20222001305 
Erick Leonel Turcios Euceda 20212021280 
Darlan Josué Perdomo Fajardo 20222000729 

# Detector de Objetos y Colores con YOLOv8 + SVM

Este proyecto es una aplicación de escritorio en Python que permite detectar objetos y predecir el color dominante en imágenes usando YOLOv8 y un clasificador SVM entrenado con características HSV y RGB.

## Características

- Detección de objetos en imágenes usando YOLOv8.
- Predicción del color dominante de los objetos detectados usando SVM.
- Interfaz gráfica amigable con Tkinter.
- Permite seleccionar imágenes o capturar desde la cámara.

## Requisitos

- Python 3.8 o superior

### Dependencias

```sh
pip install opencv-python ultralytics numpy imutils scikit-learn pillow
```

## Archivos y Carpetas

- `main.py`: Código principal de la aplicación.
- `yolov8n.pt`: Modelo pre-entrenado de YOLOv8.
- `entrenamiento/`: Carpeta con subcarpetas de imágenes de entrenamiento clasificadas por color.
- `.idea/`: Archivos de configuración del IDE (puedes ignorarlos).

## Uso

1. **Clona el repositorio:**

   ```sh
   git clone https://github.com/Elerij777/Proyecto-IA.git
   ```

2. **Prepara el dataset:**

   Asegurar que la carpeta `entrenamiento/` contenga subcarpetas con imágenes clasificadas por color (ejemplo: `Rojo/`, `Azul/`, etc.). Esto tambien servira para poder entrenar a la IA con nuevos colores

3. **Instala las dependencias:**

   ```sh
   pip install opencv-python ultralytics numpy imutils scikit-learn pillow
   ```

4. **Ejecuta la aplicación:**

   ```sh
   python main.py
   ```

5. **Usa la interfaz:**

   - Selecciona una imagen o abre la cámara.
   - Haz clic en "Detectar" para ver los resultados.
