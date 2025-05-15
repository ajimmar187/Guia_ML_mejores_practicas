


## **Mejores Prácticas en Aprendizaje Automático**

Después de trabajar en múltiples proyectos que cubren conceptos importantes de aprendizaje automático, técnicas y algoritmos ampliamente utilizados, ya tienes una visión general del ecosistema del aprendizaje automático, así como una experiencia sólida resolviendo problemas prácticos utilizando algoritmos de machine learning y Python. Sin embargo, surgirán desafíos cuando empecemos a trabajar en proyectos desde cero en el mundo real. Esta guía tiene como objetivo prepararnos para ello, con **21 mejores prácticas** para seguir a lo largo del flujo de trabajo de una solución de aprendizaje automático.

### En esta guía cubriremos los siguientes temas:

* Flujo de trabajo de una solución de aprendizaje automático
* Mejores prácticas en la etapa de preparación de datos
* Mejores prácticas en la generación del conjunto de entrenamiento
* Mejores prácticas en el entrenamiento, evaluación y selección del modelo
* Mejores prácticas en la etapa de despliegue y monitoreo

---

## **Flujo de trabajo de una solución de aprendizaje automático**

En general, las principales tareas involucradas en resolver un problema de aprendizaje automático pueden resumirse en cuatro áreas:

* Preparación de datos
* Generación del conjunto de entrenamiento
* Entrenamiento, evaluación y selección del modelo
* Despliegue y monitoreo

Desde las fuentes de datos hasta el sistema final, una solución de aprendizaje automático sigue básicamente el siguiente paradigma:

*(Aquí se muestra una figura: el ciclo de vida de una solución de aprendizaje automático)*

En las siguientes secciones, aprenderemos sobre las tareas típicas, desafíos comunes y mejores prácticas para cada una de estas cuatro etapas.

---

## **Mejores prácticas en la etapa de preparación de datos**

Ningún sistema de machine learning puede construirse sin datos. Por lo tanto, la **recolección de datos** debe ser nuestro primer enfoque.

### **Mejor práctica 1 – Comprender completamente el objetivo del proyecto**

Antes de comenzar a recolectar datos, debemos asegurarnos de que el objetivo del proyecto y el problema de negocio estén completamente entendidos. Esto nos guiará en qué fuentes de datos investigar, y requerirá también un conocimiento y experiencia de dominio suficientes.

Por ejemplo, si el objetivo es predecir los precios futuros de acciones, es crucial recopilar datos sobre el rendimiento histórico de esas acciones específicas, en lugar de datos de mercados no relacionados. Del mismo modo, para optimizar la eficiencia de campañas de publicidad en línea (medida por el ratio de clics), necesitamos recolectar datos detallados sobre los clics: información de quién hizo clic o no, en qué anuncio específico, y en qué contexto o página, en lugar de simplemente contar cuántos anuncios se mostraron.

---

### **Mejor práctica 2 – Recolectar todos los campos relevantes**

Con un objetivo definido, podemos limitar las fuentes de datos potenciales a investigar. La siguiente pregunta es: ¿es necesario recolectar todos los campos disponibles en una fuente de datos, o basta con un subconjunto?

Sería ideal saber de antemano qué atributos son clave. Sin embargo, es muy difícil asegurar que los atributos elegidos por un experto del dominio generen los mejores resultados predictivos. Por ello, se recomienda recolectar **todos los campos relacionados** al proyecto, especialmente si recoleccionar los datos más tarde es costoso o incluso imposible.

Por ejemplo, para predecir precios bursátiles, se recolectaron todos los campos disponibles: apertura, máximo, mínimo, volumen, etc., aunque al principio no sabíamos qué tan útiles serían el máximo o mínimo. En otro caso, si rascamos artículos en línea para clasificación temática, deberíamos guardar la mayor cantidad de información posible, ya que, si luego descubrimos que un campo omitido (como hipervínculos) era valioso, puede que ya no sea posible recuperar la fuente.

---

### **Mejor práctica 3 – Mantener la consistencia y normalización de valores**

En datasets existentes o recolectados, es común encontrar diferentes valores que representan lo mismo. Ejemplo: en el campo *País*, podemos ver "USA", "U.S.A", "United States"; o en *Género*, podemos encontrar "M", "Masculino", "Hombre".

Es **necesario unificar** los valores de los campos. De lo contrario, los algoritmos tratarán los mismos significados como distintos. Por ejemplo, podemos mantener solo "M", "F", y "Otro" en el campo de género, reemplazando cualquier otro valor alternativo.

También es importante asegurar la **consistencia del formato** en un campo. Por ejemplo, en el campo de edad podemos encontrar valores reales como 21, 35, y también errores como 1990, 1978. En campos de puntuación, puede haber números (1, 2) y palabras ("uno", "dos"). Estos deben transformarse a un formato uniforme.

---

### **Mejor práctica 4 – Tratar los datos faltantes**

En el mundo real, los datasets casi nunca están completamente limpios. Pueden tener valores faltantes o corruptos, como espacios en blanco, *Null*, -1, 999999, *unknown*, etc.

Estos valores no solo aportan información incompleta, sino que pueden **confundir al modelo**, que no sabe si "unknown" o -1 tienen un significado.

Hay tres estrategias básicas para tratar los valores faltantes:

1. Eliminar muestras que contienen valores faltantes
2. Eliminar campos que tengan algún valor faltante
3. **Imputar valores faltantes**: reemplazarlos por la media, mediana o el valor más frecuente del campo.

Por ejemplo, con este conjunto:
`(30, 100), (20, 50), (35, unknown), (25, 80), (30, 70), (40, 60)`

* Si eliminamos las muestras con faltantes (estrategia 1):
  `(30, 100), (20, 50), (25, 80), (30, 70), (40, 60)`
* Si eliminamos el campo completo con faltantes (estrategia 2):
  Solo queda el primer campo: 30, 20, 35, 25, 30, 40
* Si imputamos el valor faltante con la **media** del campo:
  `(35, 72)` — suponiendo que la media es 72
  


Python ofrece herramientas como `SimpleImputer` en Scikit-learn para realizar imputación automáticamente.

---

### **Mejor práctica 5 – Almacenamiento de datos a gran escala**

Con el crecimiento exponencial del volumen de datos, muchas veces no podemos almacenar todo en una sola máquina local. Por eso, necesitamos recurrir al **almacenamiento en la nube** o sistemas de archivos distribuidos.

Existen dos estrategias principales para escalar el almacenamiento:

* **Escalado vertical (scale-up):** consiste en aumentar la capacidad del sistema actual, por ejemplo, añadiendo más discos. Es útil cuando se requiere acceso rápido.
* **Escalado horizontal (scale-out):** se incrementa la capacidad añadiendo nuevos nodos a un clúster. Sistemas como **HDFS** (Hadoop Distributed File System) o **Spark** se utilizan para distribuir los datos entre cientos o miles de nodos.

También existen servicios de almacenamiento en la nube, como:

* **Amazon S3** (AWS)
* **Google Cloud Storage**
* **Microsoft Azure Storage**

Además del sistema de almacenamiento, se deben considerar las siguientes prácticas:

* **Particionado de datos:** dividir en fragmentos pequeños para distribuir la carga.
* **Compresión y codificación:** reducir espacio de almacenamiento y tiempos de recuperación.
* **Replicación:** duplicar datos en distintos nodos o ubicaciones para tolerancia a fallos.
* **Seguridad y control de acceso:** asegurar que solo usuarios autorizados accedan a los datos.

Una vez que los datos están bien preparados, podemos pasar a la **generación del conjunto de entrenamiento**.

---

## **Mejores prácticas en la etapa de generación del conjunto de entrenamiento**

Las tareas típicas en esta etapa se agrupan en dos categorías:

1. **Preprocesamiento de datos**
2. **Ingeniería de características (features)**

---

### **Mejor práctica 6 – Identificar variables categóricas con valores numéricos**

Las variables categóricas suelen ser obvias (nivel de riesgo, ocupación, intereses), pero a veces pueden parecer numéricas, como:

* 1 a 12 (meses del año)
* 0 y 1 (falso/verdadero)

**¿Cómo diferenciarlas?**
Si implican una relación matemática o de orden, son numéricas (por ejemplo: calificación de 1 a 5). Si no, son categóricas (meses, días de la semana, etc.).

---

### **Mejor práctica 7 – Decidir si codificar variables categóricas**

Si una característica es categórica, hay que decidir si codificarla o no, dependiendo del algoritmo:

* **Naïve Bayes y algoritmos basados en árboles** pueden usar variables categóricas directamente.
* Otros algoritmos (regresión, SVM, redes neuronales) **requieren codificación**, como *one-hot encoding* o *label encoding*.

Es clave ver las etapas de **generación de características** y **entrenamiento del modelo** como un conjunto, no como procesos separados.

---

### **Mejor práctica 8 – Decidir si seleccionar características, y cómo hacerlo**

Seleccionar características puede reducir tiempo de entrenamiento, evitar sobreajuste y mejorar rendimiento.
Ejemplos: regresión logística con regularización L1, random forest.

Sin embargo, no siempre mejora la precisión, por lo que **es recomendable comparar** resultados con y sin selección usando validación cruzada.

Un ejemplo en Scikit-learn con el dataset de dígitos escritos a mano muestra que usando solo las 25 características más importantes (de 64), la precisión del modelo SVM mejora de 0.90 a 0.95.

---

### **Mejor práctica 9 – Decidir si reducir la dimensionalidad, y cómo hacerlo**

A diferencia de la selección de características, la reducción de dimensionalidad transforma las variables originales a un nuevo espacio (por ejemplo, PCA).

Ventajas:

* Reduce tiempo de entrenamiento
* Disminuye sobreajuste
* Puede mejorar el rendimiento

Al igual que antes, no garantiza mejor desempeño. Debe evaluarse con validación cruzada. En el mismo dataset de dígitos, reducir a las 15 componentes principales con PCA también mejora el rendimiento a 0.95.

---

### **Mejor práctica 10 – Decidir si escalar las características**

Modelos como regresión lineal con SGD, SVR y redes neuronales **requieren** que las características estén estandarizadas (media cero, varianza unitaria).

**¿Cuándo no es necesario?**

* Naïve Bayes y árboles de decisión no son sensibles a escalas distintas.

**¿Cuándo sí es necesario?**

* Algoritmos que utilizan distancias o separación en el espacio (SVC, SVR, KNN, K-means)
* Algoritmos que usan descenso de gradiente (regresión/logística, redes neuronales)



---

## **Mejor práctica 11 – Realizar ingeniería de características con conocimiento del dominio**

Si contamos con conocimiento del dominio, podemos crear características específicas que se alineen con el negocio y el problema.
Por ejemplo, al predecir precios bursátiles, podemos diseñar características basadas en factores que los inversionistas suelen considerar, como el volumen de transacciones o las variaciones diarias.

También hay prácticas generales aplicables sin importar el dominio. En marketing o análisis de clientes, **el momento del día, día de la semana o mes** suelen ser señales importantes.
Dado un dato con la fecha `2020/09/01` y hora `14:34:21`, podríamos generar características como: `tarde`, `martes`, `septiembre`.

En comercio minorista, también es útil **agrupar información temporalmente**:
Ejemplo:

* Total de visitas en los últimos tres meses
* Promedio de productos comprados semanalmente el año anterior

Estas son buenas predicciones del comportamiento futuro del cliente.

---

## **Mejor práctica 12 – Realizar ingeniería de características sin conocimiento del dominio**

¿Y si no tenemos conocimiento específico? No te preocupes. Hay enfoques **genéricos** que puedes aplicar:

### **Binarización y discretización**

**Binarización:** transforma una característica numérica en binaria con un umbral.
Ejemplo: si el término “premio” aparece más de una vez en un correo, lo codificamos como 1, si no, como 0.

```python
from sklearn.preprocessing import Binarizer
X = [[4], [1], [3], [0]]
binarizer = Binarizer(threshold=2.9)
X_new = binarizer.fit_transform(X)
# Resultado: [[1], [0], [1], [0]]
```

**Discretización:** convierte un número en categorías.
Ejemplo: para el campo edad podríamos crear grupos:

* 18–24
* 25–34
* 35–54
* 55+

---

### **Interacción entre características**

Crear nuevas características combinando otras:

* Numéricas: suma, producto, etc.

  * Ej: visitas por semana × productos comprados por semana → productos por visita.
* Categóricas: combinación conjunta

  * Ej: profesión e interés → "ingeniero deportista"

---

### **Transformación polinómica**

Genera nuevas características mediante potencias e interacciones entre variables.

Ejemplo con Scikit-learn:

```python
from sklearn.preprocessing import PolynomialFeatures

X = [[2, 4], [1, 3], [3, 2], [0, 3]]
poly = PolynomialFeatures(degree=2)
X_new = poly.fit_transform(X)
```

Esto genera:

* 1 (intercepto)
* a, b, a², ab, b²

---

## **Mejor práctica 13 – Documentar cómo se generó cada característica**

Puede parecer trivial, pero muchas veces se nos olvida cómo se creó una característica.
Esto se vuelve importante si el modelo falla y necesitamos regresar a crear nuevas variables o eliminar las que no funcionaron.

**Registrar cada paso** permite modificar, reproducir y mejorar sin perder contexto.

---

## **Mejor práctica 14 – Extraer características de texto**

Hay dos enfoques principales:

### **1. Enfoques tradicionales: TF y TF-IDF**

* **TF (Term Frequency):** cuenta cuántas veces aparece un término.
* **TF-IDF:** ajusta el conteo penalizando términos comunes y destacando los raros.

Esto se conoce como **Bolsa de Palabras (BoW)**, y no considera el orden de los términos.
Desventaja: genera vectores dispersos, de alta dimensionalidad y sin contexto semántico.

---

### **2. Word Embedding (incrustaciones de palabras)**

A diferencia de TF/TF-IDF, word embedding representa cada palabra como un vector **denso de valores reales**.
Estos vectores capturan **significados y relaciones semánticas**.
Ejemplo: los vectores de "clustering" y "grouping" estarán cerca si se usan en contextos similares.

#### **Formas de obtener embeddings:**

* **Entrenar con Word2Vec** usando *Skip-gram* o *CBOW*
* **Usar embeddings preentrenados** como:

  * **FastText**
  * **GloVe**
  * **BERT**
  * **GPT**
  * **USE (Universal Sentence Encoder)**

#### **Ejemplo simple con Gensim (Word2Vec):**

```python
from gensim.models import Word2Vec

sentences = [
    ["i", "love", "machine", "learning", "by", "example"],
    ["machine", "learning", "and", "deep", "learning", "are", "fascinating"]
]

model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0)
vector = model.wv["machine"]
```

---

#### **Embeddings en redes neuronales**

En redes neuronales modernas (especialmente en NLP), se usa una **capa de embedding** que aprende automáticamente las representaciones.

**Ejemplo en PyTorch:**

```python
import torch
import torch.nn as nn

input_data = torch.LongTensor([[1, 2, 3, 4], [5, 1, 6, 3]])
embedding_layer = nn.Embedding(10, 3)
embedded_data = embedding_layer(input_data)
```

Esto convierte índices de palabras en vectores.

---

## **Mejores prácticas en la etapa de entrenamiento, evaluación y selección del modelo**

Al abordar un problema supervisado, muchos se preguntan de inmediato:
**¿Cuál es el mejor algoritmo para clasificar o hacer regresión?**
No hay una respuesta universal. No existe una solución mágica. No sabrás cuál algoritmo funciona mejor **hasta probar varios** y ajustar sus parámetros.

---

### **Mejor práctica 15 – Elegir el(los) algoritmo(s) inicial(es)**

Probar todos los algoritmos y ajustarlos es costoso. En vez de eso, selecciona de **1 a 3 algoritmos candidatos** utilizando estas consideraciones:

* Tamaño del dataset
* Dimensionalidad de los datos
* ¿Los datos son linealmente separables?
* ¿Las características son independientes?
* Tolerancia al sesgo/varianza
* ¿Se necesita aprendizaje en línea?

#### **Guía breve por algoritmo:**

##### **Naïve Bayes**

* Ideal si las características son independientes.
* Funciona bien incluso con pocos datos.
* Entrena muy rápido.
* Alto sesgo, baja varianza.

##### **Regresión logística**

* Muy utilizada, ideal cuando los datos son (aproximadamente) linealmente separables.
* Escalable a datasets grandes con descenso de gradiente estocástico (SGD).
* Soporta aprendizaje en línea.
* Riesgo de sobreajuste mitigado con regularización L1/L2.

##### **SVM (Máquinas de Vectores de Soporte)**

* Adaptable a separación lineal o no lineal (mediante kernels).
* Excelente para datos de alta dimensionalidad (ej.: clasificación de textos).
* Precisa, pero computacionalmente exigente.

##### **Árboles de decisión / Random Forest**

* No importa la separación lineal.
* Acepta variables categóricas sin codificación.
* Modelo interpretable y explicable.
* Random Forest reduce sobreajuste mediante el ensamblado de árboles.

##### **Redes neuronales**

* Muy potentes, especialmente en visión por computadora y procesamiento de texto.
* Difíciles de ajustar (capas, nodos, funciones de activación, etc.).
* Requieren muchos datos.
* Costosas en tiempo de entrenamiento.

---

### **Mejor práctica 16 – Reducir el sobreajuste**

Recapitulando estrategias clave:

* **Más datos**: ayuda a evitar que el modelo aprenda ruido.
* **Simplificar el modelo**: evita complejidad innecesaria.
* **Validación cruzada**: buena práctica estándar.
* **Regularización**: penaliza la complejidad (L1, L2).
* **Early stopping**: detener el entrenamiento cuando el rendimiento en validación se degrada.
* **Dropout** (en redes): desconecta aleatoriamente neuronas durante el entrenamiento.
* **Selección de características**: eliminar atributos irrelevantes.
* **Ensamblado**: combinar modelos simples (bagging, boosting).

---

### **Mejor práctica 17 – Diagnosticar sobreajuste y subajuste**

Usamos **curvas de aprendizaje** para analizar bias y varianza. Se comparan los errores en entrenamiento vs validación con el número de muestras.

* **Sobreajuste**: alto rendimiento en entrenamiento, bajo en validación.
* **Subajuste**: ambos rendimientos bajos.
* **Ideal**: las curvas convergen con rendimiento alto.

Scikit-learn proporciona el módulo `learning_curve` para visualizar estos gráficos y diagnosticar problemas.

---

### **Mejor práctica 18 – Modelar datasets a gran escala**

Trabajar con grandes volúmenes requiere estrategia:

#### **Consejos clave:**

* **Empieza con un subconjunto pequeño**: para experimentar rápidamente.
* **Usa algoritmos escalables**: regresión logística, SVM lineal, SGD.
* **Computación distribuida**: frameworks como Apache Spark.
* **Reducción de dimensionalidad**: PCA, t-SNE si es necesario.
* **Paralelización**: usar múltiples GPUs o nodos.
* **Administración de memoria**: carga por lotes, liberación eficiente.
* **Bibliotecas optimizadas**: como TensorFlow, PyTorch, XGBoost.
* **Aprendizaje incremental**: para datos en streaming o que llegan progresivamente.

> ⚠️ ¡No olvides guardar el modelo entrenado! Entrenar con datos grandes toma tiempo y recursos.

---

## **Mejores prácticas en la etapa de despliegue y monitoreo**

Después de preparar los datos, generar el conjunto de entrenamiento y entrenar el modelo, llega el momento de **desplegar el sistema**. Aquí nos aseguramos de que los modelos funcionen bien en producción, se actualicen si es necesario y sigan ofreciendo valor real.

---

### **Mejor práctica 19 – Guardar, cargar y reutilizar modelos**

Al desplegar un modelo, los nuevos datos deben pasar por el **mismo proceso de preprocesamiento** que se usó en el entrenamiento: escalado, ingeniería de características, selección, etc.

Por eso, **no se debe reentrenar todo el pipeline desde cero cada vez**. En cambio, se deben guardar:

* El modelo de preprocesamiento (escaladores, transformadores)
* El modelo entrenado

Y luego cargarlos cuando se necesiten para hacer predicciones.

#### **Guardar con `joblib` (Scikit-learn)**

```python
from joblib import dump, load

# Guardar el escalador
dump(scaler, "scaler.joblib")

# Guardar el modelo
dump(regressor, "regressor.joblib")
```

Luego, en producción:

```python
# Cargar los objetos
scaler = load("scaler.joblib")
regressor = load("regressor.joblib")

# Preprocesar y predecir
X_scaled = scaler.transform(X_new)
predicciones = regressor.predict(X_scaled)
```

Joblib es más eficiente que pickle para objetos de NumPy y modelos de machine learning, especialmente con datasets grandes, y ofrece mejor compresión y rendimiento.

---

#### **Guardar modelos en TensorFlow**

```python
# Guardar modelo completo
model.save('./modelo_tf')

# Cargar modelo
nuevo_modelo = tf.keras.models.load_model('./modelo_tf')
```

---

#### **Guardar modelos en PyTorch**

```python
# Guardar modelo completo
torch.save(model, './modelo.pth')

# Cargar modelo
nuevo_modelo = torch.load('./modelo.pth')
```

Esto guarda arquitectura, pesos y configuración del entrenamiento.

---

### **Mejor práctica 20 – Monitorear el rendimiento del modelo**

Una vez desplegado el modelo, **debe ser monitoreado continuamente** para asegurarse de que siga funcionando bien. Algunos consejos:

* **Define métricas claras**: precisión, F1, AUC-ROC, R², error cuadrático medio, etc.
* **Compara contra un modelo base** (baseline): útil como referencia.
* **Curvas de aprendizaje**: visualizan si hay sobreajuste o subajuste.

Ejemplo en Scikit-learn:

```python
from sklearn.metrics import r2_score
print(f'Chequeo del modelo, R^2: {r2_score(y_nuevo, predicciones):.3f}')
```

Además, deberías registrar (loggear) estas métricas y **activar alertas** si el rendimiento baja.

---

### **Mejor práctica 21 – Actualizar los modelos regularmente**

Con el tiempo, los datos pueden cambiar (fenómeno conocido como *data drift*). Si el rendimiento se deteriora:

* **Monitorea constantemente**: si las métricas bajan, es momento de actuar.
* **Actualizaciones programadas**: según frecuencia de cambios en los datos.
* **Aprendizaje en línea (online learning)**: para modelos como regresión con SGD o Naïve Bayes, que se pueden actualizar sin reentrenar.
* **Control de versiones**: tanto de modelos como de datasets.
* **Auditorías regulares**: revisa si las métricas, objetivos de negocio o datos han cambiado.

> 📌 Monitorear es un proceso continuo, no algo que se hace una sola vez.

---

## **Resumen**

Esta guía te prepara para resolver problemas reales de machine learning. Repasamos el flujo de trabajo típico:

1. Preparación de datos
2. Generación del conjunto de entrenamiento
3. Entrenamiento, evaluación y selección de modelos
4. Despliegue y monitoreo

Para cada etapa, detallamos tareas, desafíos comunes y **21 mejores prácticas**.

> ✅ **La mejor práctica de todas es practicar.**
> Empieza un proyecto real y aplica lo que has aprendido.