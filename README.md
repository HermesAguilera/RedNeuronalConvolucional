# Red Neuronal Convolucional

## Descripción del proyecto
Este proyecto implementa una Red Neuronal Convolucional (CNN) desde cero en Python para clasificar imágenes en 5 clases distintas y detectar si una imagen no pertenece a ninguna de ellas.

El código está desarrollado sin librerías de alto nivel para redes neuronales como TensorFlow, Keras, PyTorch o Caffe. Solo utiliza librerías básicas para el manejo numérico (`numpy`) y de imágenes (`PIL`).

## Estructura del repositorio
- `main.py` - pipeline principal de carga de datos, entrenamiento, evaluación y predicción.
- `model.py` - definición del modelo CNN, forward y backpropagation.
- `layers.py` - implementación de capas convolucionales, pooling y densas.
- `activations.py` - implementaciones de ReLU y Softmax con sus derivadas.
- `utils.py` - carga y preprocesamiento de datos, funciones auxiliares, split de datos.
- `cnn.ipynb` - notebook para experimentación y visualización de resultados.

## Arquitectura de la CNN
La red incluye:
- 3 capas convolucionales
- 3 capas de activación ReLU
- 3 capas de pooling (MaxPooling)
- 2 capas densas antes de la salida
- capa de salida con `Softmax`

La arquitectura general es:
- Entrada `64x64` en RGB
- Conv(8 filtros, 3x3) + ReLU + MaxPool
- Conv(16 filtros, 3x3) + ReLU + MaxPool
- Conv(32 filtros, 3x3) + ReLU + MaxPool
- Dense 32 -> ReLU
- Dense `num_classes` -> Softmax

### Justificación de la arquitectura
- Tener varias capas convolucionales permite extraer características jerárquicas: bordes, texturas y formas.
- El pooling reduce la dimensionalidad y agrega tolerancia a pequeñas variaciones en las imágenes.
- Una capa totalmente conectada antes de la salida es necesaria para combinar las características extraídas en una predicción final.
- La salida `Softmax` proporciona una distribución de probabilidad sobre las clases.

## Backpropagation y entrenamiento
Se implementa explícitamente:
- `forward` para propagar datos desde la entrada hasta la salida.
- `backward` para propagar errores desde la salida hacia las capas anteriores.
- gradientes individuales para cada tipo de capa:
  - `ConvolutionalLayer.backward`
  - `PoolingLayer.backward`
  - `DenseLayer.backward`
- actualización de pesos con SGD simple dentro de cada capa.

### Justificación técnica
- La exigencia de programar backprop desde cero se cumple con las derivadas de cada operación.
- Usar un optimizador simple como SGD es suficiente y adecuado para un proyecto académico de CNN desde cero.

## Funciones de activación
Se usan las siguientes funciones propias:
- `ReLU` (Rectified Linear Unit)
- `Softmax`

### Por qué ReLU
- Es simple y eficiente.
- Permite evitar saturación de gradientes para valores positivos.
- Es una función estándar en CNN modernas.

### Por qué Softmax
- Convierte las salidas en probabilidades.
- Permite elegir la clase más probable y aplicar un umbral de confianza para rechazo.

## Detección de "ninguna de las anteriores"
Se aplica un umbral de confianza en la salida de `Softmax`.
- Si la probabilidad máxima es menor que el umbral, la predicción retorna `"Ninguna de las anteriores"`.
- El umbral actual está configurado en `0.25`.

### Justificación
- Este método es simple y efectivo para detectar casos fuera de las clases entrenadas.
- No requiere un modelo adicional ni técnicas de ensamblado complejas.

## Augmentación de datos
Se implementa una transformación simple antes del entrenamiento:
- Flip horizontal aleatorio sobre el batch de entrenamiento.

### Justificación
- Aumenta la variabilidad del dataset sin recopilar más imágenes.
- Mejora la generalización del modelo ante diferentes orientaciones.

## Inicialización y optimización
- `learning_rate = 0.15`
- `batch_size = 32`
- pesos inicializados con `0.1 * np.random.randn(...)`

### Justificación
- El aprendizaje con una tasa mayor acelera la convergencia en un modelo pequeño y con pocos datos.
- La inicialización no nula evita simetrías y ayuda al entrenamiento.
- `batch_size = 32` es un valor estándar que equilibra estabilidad y velocidad.
