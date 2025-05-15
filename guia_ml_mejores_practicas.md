


# **Guía Completa de Mejores Prácticas en Aprendizaje Automático**

> *"En la teoría, no hay diferencia entre teoría y práctica. En la práctica, sí la hay."* - Yogi Berra

## **Introducción**

¡Bienvenido a esta guía práctica de Machine Learning! 

Después de trabajar en múltiples proyectos que cubren conceptos importantes de aprendizaje automático, técnicas y algoritmos ampliamente utilizados, ya tienes una visión general del ecosistema del aprendizaje automático, así como una experiencia sólida resolviendo problemas prácticos utilizando algoritmos de machine learning y Python. 

Sin embargo, cuando pasamos de los ejemplos académicos a **proyectos reales**, surgen desafíos inesperados que no suelen abordarse en los cursos tradicionales:

- ¿Cómo lidiar con datos incompletos o inconsistentes?
- ¿Qué hacer cuando el modelo funciona bien en desarrollo pero falla en producción?
- ¿Cómo determinar qué algoritmo elegir entre tantas opciones?

Esta guía tiene como objetivo prepararte para estos escenarios con **21 mejores prácticas** esenciales que debes seguir a lo largo del ciclo de vida completo de una solución de aprendizaje automático. Cada práctica viene respaldada por ejemplos concretos y código de implementación.

### **En esta guía cubriremos:**

📊 **Flujo de trabajo completo** de una solución de aprendizaje automático  
🔍 **Mejores prácticas en preparación de datos** - el fundamento de todo buen modelo  
⚙️ **Técnicas de generación de conjuntos de entrenamiento** óptimos  
🧠 **Estrategias para entrenamiento, evaluación y selección** de modelos  
🚀 **Métodos efectivos de despliegue y monitoreo** en entornos productivos

> **Nota para estudiantes:** Al final de cada sección encontrarás ejercicios prácticos y preguntas de reflexión para afianzar lo aprendido.

---

## **1. Flujo de trabajo de una solución de aprendizaje automático**

Cuando abordamos un proyecto real de Machine Learning, seguimos un flujo de trabajo estructurado que puede dividirse en cuatro grandes etapas:

![Ciclo de vida ML](https://i.imgur.com/placeholder_ml_workflow.png)

| Etapa | Descripción | Objetivo principal |
|-------|-------------|-------------------|
| **1. Preparación de datos** | Recolección, limpieza y estructuración | Obtener datos limpios y representativos |
| **2. Generación del conjunto de entrenamiento** | Preprocesamiento e ingeniería de características | Transformar datos crudos en features predictivas |
| **3. Entrenamiento y evaluación** | Construcción, validación y selección de modelos | Obtener el mejor modelo posible |
| **4. Despliegue y monitoreo** | Implementación, seguimiento y mantenimiento | Mantener el modelo funcionando correctamente |

Este ciclo no es lineal sino **iterativo** - los resultados de cada etapa pueden llevarnos a revisitar etapas anteriores para realizar ajustes.

> **Reflexión:** Antes de continuar, piensa en algún proyecto de ML que hayas realizado. ¿Seguiste conscientemente estas etapas? ¿Cuál te resultó más desafiante?

Analicemos ahora las mejores prácticas para cada etapa, comenzando con la preparación de datos, el fundamento de todo buen modelo.

---

## **2. Mejores prácticas en la etapa de preparación de datos**

> *"Si la basura entra, la basura sale."* - Principio fundamental en ciencia de datos

Ningún sistema de machine learning, por sofisticado que sea, puede superar las limitaciones de datos deficientes. La **calidad de los datos** es el factor más determinante en el éxito de un proyecto. Por ello, la **recolección y preparación de datos** debe ser nuestra primera prioridad.

### **2.1. Comprender profundamente el objetivo del proyecto**

**Problema:** Frecuentemente nos apresuramos a recolectar datos sin entender completamente lo que intentamos resolver.

**Solución:** Antes de escribir una sola línea de código, debemos:
- Formular claramente el problema de negocio
- Definir métricas de éxito concretas
- Entender el contexto y las restricciones del problema
- Consultar con expertos del dominio

#### **Ejemplo práctico:**

| Objetivo mal definido | Objetivo bien definido |
|-----------------------|-----------------------|
| "Predecir precios de acciones" | "Predecir el precio de cierre diario de la acción XYZ con ±2% de error, usando datos históricos de los últimos 5 años" |
| "Mejorar campañas de marketing" | "Aumentar la tasa de conversión de clics (CTR) en un 15% identificando qué características de los anuncios generan más interacción" |

> **Ejercicio:** Para un proyecto que te interese, escribe primero el objetivo en términos generales y luego refínalo hasta que sea específico, medible y accionable.

---

### **2.2. Recolectar todos los campos potencialmente relevantes**

**Problema:** A menudo limitamos la recolección a campos que inicialmente parecen relevantes, solo para descubrir después que necesitamos datos adicionales que ya no podemos recuperar.

**Solución:** Adoptar una estrategia más exhaustiva:

- Recolectar todos los campos relacionados con el dominio del problema
- Documentar metadatos (origen, timestamp, procesos de extracción)
- Priorizar la completitud sobre la eficiencia inicial

#### **Consideraciones prácticas:**

```python
# Ejemplo: Recolección completa vs. parcial para predicción bursátil
# Enfoque limitado (solo lo que creemos necesario)
df_limitado = api.get_stock_data(symbol='AAPL', fields=['date', 'close_price'])

# Enfoque exhaustivo (todos los campos disponibles)
df_completo = api.get_stock_data(symbol='AAPL', 
                                fields=['date', 'open', 'high', 'low', 
                                       'close', 'volume', 'adj_close',
                                       'dividends', 'splits', 'market_cap'])

# El análisis posterior podría revelar que 'volume' es un predictor clave,
# algo que habríamos perdido con el enfoque limitado
```

> **Nota:** En el caso de web scraping o extracción de datos de fuentes volátiles, es particularmente importante guardar todos los datos posibles, ya que es posible que no puedas volver a acceder a la fuente original.

#### **Costo-beneficio de la recolección exhaustiva:**

| Ventajas | Desventajas | Estrategia de mitigación |
|----------|-------------|--------------------------|
| No perder variables predictivas importantes | Mayor costo de almacenamiento | Comprimir datos o usar formatos eficientes (Parquet, HDF5) |
| Habilitar análisis exploratorio más completo | Procesamiento inicial más lento | Muestrear para análisis exploratorio inicial |
| Poder responder nuevas preguntas en el futuro | Potencial sobrecarga de información | Documentar bien todos los campos para facilitar su uso |

---

### **2.3. Estandarizar y normalizar valores consistentemente**

**Problema:** Los datos del mundo real presentan inconsistencias que los algoritmos no pueden interpretar correctamente: "USA" vs "U.S.A" vs "Estados Unidos", formatos de fecha diferentes, o valores numéricos con distintas unidades.

**Solución:** Implementar un proceso sistemático de estandarización:

1. Identificar campos problemáticos mediante análisis exploratorio
2. Crear diccionarios de mapeo para valores equivalentes
3. Aplicar transformaciones consistentes

#### **Ejemplo práctico: Normalización de países**

```python
# Diccionario de normalización
pais_normalizacion = {
    'USA': 'United States',
    'U.S.A.': 'United States',
    'United States of America': 'United States',
    'Estados Unidos': 'United States',
    'US': 'United States',
    'América': 'United States',
    # ... más variaciones
}

# Aplicar normalización
df['pais_normalizado'] = df['pais'].replace(pais_normalizacion)
```

#### **Herramientas avanzadas:**

Bibliotecas como **pandas-dedupe** o **recordlinkage** ofrecen capacidades de coincidencia difusa para casos más complejos:

```python
import recordlinkage as rl
from recordlinkage.preprocessing import clean

# Limpieza básica
df['pais_limpio'] = clean(df['pais'])

# Comparación usando similitud de cadenas
indexer = rl.Index()
indexer.block('pais_limpio')
candidatos = indexer.index(df)

comparador = rl.Compare()
comparador.string('pais_limpio', 'pais_limpio', method='jarowinkler', threshold=0.85)
coincidencias = comparador.compute(candidatos, df)
```

> **Buena práctica**: Automatiza este proceso pero mantén un registro de las transformaciones realizadas para poder revertirlas si es necesario.

---

### **2.4. Tratar estratégicamente los datos faltantes**

**Problema:** Los datasets reales casi siempre tienen valores faltantes (NaN, NULL, espacios en blanco, -1, 999999, etc.) que pueden sesgar significativamente los resultados del modelo.

**Solución:** Adoptar un enfoque sistemático basado en:
- El tipo de datos
- El mecanismo de ausencia (MCAR, MAR, MNAR)¹
- El porcentaje de valores faltantes
- La importancia de la variable

#### **Estrategias de tratamiento:**

| Estrategia | Cuándo usarla | Ventajas | Desventajas |
|------------|---------------|----------|-------------|
| **1. Eliminar registros** | Pocas filas afectadas (<5%) | Simple y rápido | Pérdida de información potencialmente valiosa |
| **2. Eliminar variables** | Alto % de valores faltantes (>50%) | Reduce ruido | Pérdida de características potencialmente predictivas |
| **3. Imputación simple** | Volumen moderado (5-20%) | Preserva todos los datos | Puede introducir sesgos |
| **4. Imputación avanzada** | Valores importantes pero faltantes | Mayor precisión | Mayor complejidad |

_¹ MCAR (Missing Completely at Random), MAR (Missing at Random), MNAR (Missing Not at Random)_

#### **Ejemplo con Scikit-learn:**

```python
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
import pandas as pd

# Dataset de ejemplo con valores faltantes
data = np.array([
    [30, 100], [20, 50], [35, np.nan],
    [25, 80], [30, 70], [40, 60]
])
df = pd.DataFrame(data, columns=['edad', 'peso'])

# 1. Imputación simple (media, mediana, moda)
imputer_media = SimpleImputer(strategy='mean')
data_imputada_media = imputer_media.fit_transform(df)

# 2. Imputación KNN (basada en vecinos cercanos)
imputer_knn = KNNImputer(n_neighbors=2)
data_imputada_knn = imputer_knn.fit_transform(df)

# Comparación de resultados
print("Original con faltantes:\n", df)
print("\nImputación con media:\n", pd.DataFrame(data_imputada_media, columns=df.columns))
print("\nImputación con KNN:\n", pd.DataFrame(data_imputada_knn, columns=df.columns))
```

#### **Métodos avanzados de imputación:**

- **MICE** (Multiple Imputation by Chained Equations)
- **Imputación basada en modelos** (usando árboles de decisión o KNN)
- **Imputación con redes neuronales** (autoencoders)
- **Análisis de sensibilidad** para evaluar el impacto de diferentes métodos

> **Consejo práctico:** Crea una columna indicadora para cada variable con muchos valores faltantes. Esta "bandera de ausencia" puede ser en sí misma una característica predictiva importante.

---

### **2.5. Implementar estrategias eficientes para datos a gran escala**

**Problema:** El volumen de datos puede crecer rápidamente hasta superar la capacidad de procesamiento y almacenamiento de una sola máquina.

**Solución:** Adoptar arquitecturas y tecnologías diseñadas específicamente para manejar grandes volúmenes de datos.

#### **Estrategias de escalado principales:**

![Comparación escalado vertical vs horizontal](https://i.imgur.com/placeholder_scaling.png)

| Estrategia | Descripción | Casos de uso ideales |
|------------|-------------|----------------------|
| **Escalado vertical** | Aumentar capacidad de una sola máquina (más RAM, CPU, SSD) | • Datasets medianos (hasta ~100GB)<br>• Análisis que requieren baja latencia<br>• Operaciones que no se paralelizan bien |
| **Escalado horizontal** | Distribuir datos y procesamiento entre múltiples nodos | • Datasets masivos (TB, PB)<br>• Procesamiento batch<br>• Tareas paralelizables |

#### **Tecnologías recomendadas por escenario:**

**Para almacenamiento:**
```
• Datasets pequeños (<10GB): Archivos CSV, SQLite
• Datasets medianos (10GB-100GB): PostgreSQL, MySQL, HDF5, Parquet
• Datasets grandes (>100GB): 
  - Cloud: Amazon S3, Google Cloud Storage, Azure Blob Storage
  - On-premise: HDFS, Ceph, MinIO
```

**Para procesamiento:**
```
• Datasets pequeños/medianos: pandas, NumPy, scikit-learn
• Datasets grandes: 
  - Spark (PySpark para Python)
  - Dask (alternativa en Python puro)
  - Ray (para ML distribuido)
```

#### **Ejemplo de código con Dask (alternativa a pandas para datos grandes):**

```python
import dask.dataframe as dd

# Crear un DataFrame Dask a partir de múltiples archivos CSV
# Soporta wildcard para cargar muchos archivos a la vez
df = dd.read_csv('datos_*.csv')

# Las operaciones son perezosas (lazy) - no se ejecutan hasta que se necesitan
result = df.groupby('categoria').agg({'ventas': 'sum'})

# Visualizar solo cuando sea necesario
print(result.compute())  # Ahora se realizan los cálculos
```

#### **Consideraciones adicionales esenciales:**

- **Particionado inteligente:** Divide los datos por fechas, regiones u otras dimensiones lógicas
- **Formatos optimizados:** Prioriza formatos columna (Parquet, ORC) sobre formatos fila (CSV)
- **Compresión adecuada:** Utiliza algoritmos que permitan lectura parcial (Snappy, LZ4) 
- **Caché y materialización:** Guarda resultados intermedios para evitar recalcular
- **Estrategias de muestreo:** Trabaja con muestras representativas para desarrollo

> **Pregunta para reflexionar:** ¿Cómo cambiaría tu enfoque si tus datos actuales crecieran 100 veces en volumen?

---

## **3. Mejores prácticas en la generación del conjunto de entrenamiento**

> *"Los datos no siempre cuentan historias verdaderas; depende de cómo los preparemos para que hablen."*

Una vez que tenemos datos limpios y consistentes, llega el momento crítico de transformarlos en información que nuestros algoritmos puedan aprovechar al máximo. Esta etapa determina en gran medida el rendimiento final de nuestros modelos.

Las tareas en esta fase se pueden agrupar en dos categorías principales:

1. **Preprocesamiento de datos:** transformaciones necesarias para que los algoritmos puedan operar correctamente
2. **Ingeniería de características (feature engineering):** creación de variables predictivas a partir de los datos crudos

![Flujo de ingeniería de características](https://i.imgur.com/placeholder_feature_eng.png)

Analicemos las mejores prácticas para esta etapa crucial:

### **3.1. Identificar correctamente variables categóricas con apariencia numérica**

**Problema:** Algunas variables parecen numéricas pero realmente representan categorías, y tratarlas incorrectamente afecta al modelo.

**Solución:** Analizar la naturaleza semántica de cada variable y no solo su tipo de datos.

#### **Guía para identificación:**

| Característica | Variable numérica | Variable categórica |
|----------------|-------------------|---------------------|
| **Operaciones matemáticas** | Tienen sentido (ej: edad+2) | No tienen sentido (ej: mes+2) |
| **Cardinalidad** | Generalmente alta | Generalmente limitada |
| **Valor semántico** | Magnitud importante | Solo la categoría importa |
| **Ejemplos** | Edad, ingresos, altura | Códigos postales, meses, IDs |

#### **Casos comúnmente confusos:**

```
• Valores 0/1: ¿Son binarios (numéricos) o dos categorías?
• Rangos 1-5: ¿Son calificaciones ordinales o valores continuos?
• Años: ¿Importa su valor numérico o son categorías temporales?
• Códigos numéricos: ¿El orden o magnitud tiene sentido?
```

#### **Ejemplo de diagnóstico en Python:**

```python
def diagnosticar_variable(serie):
    """Analiza una variable y sugiere su posible tipo"""
    n_valores_unicos = serie.nunique()
    
    # Verificar proporciones y patrones
    proporcion_valores_unicos = n_valores_unicos / len(serie)
    tiene_valores_fraccionales = (serie % 1 != 0).any()
    rango = serie.max() - serie.min()
    
    print(f"Valores únicos: {n_valores_unicos} ({proporcion_valores_unicos:.2%} del total)")
    print(f"Tiene valores fraccionales: {tiene_valores_fraccionales}")
    print(f"Rango: {rango}")
    
    # Sugerencia basada en heurísticas
    if n_valores_unicos <= 20:
        if rango < 10 and not tiene_valores_fraccionales:
            return "Probablemente categórica"
    
    return "Probablemente numérica"
```

> **Consejo práctico:** Cuando tengas dudas, prueba modelos con ambos enfoques (tratando la variable como categórica y como numérica) y compara resultados.

---

### **3.2. Aplicar la codificación adecuada para variables categóricas**

**Problema:** Diferentes algoritmos tienen distintos requisitos para procesar variables categóricas, y una codificación incorrecta puede degradar el rendimiento.

**Solución:** Seleccionar la técnica de codificación según el algoritmo y las características específicas de los datos.

#### **Técnicas de codificación principales:**

| Técnica | Descripción | Mejor para | Limitaciones |
|---------|-------------|------------|--------------|
| **Label Encoding** | Asigna un número entero a cada categoría | • Árboles de decisión<br>• Algoritmos que pueden manejar relaciones ordinales | Introduce orden artificial entre categorías |
| **One-Hot Encoding** | Crea una columna binaria por cada categoría | • Regresión<br>• SVM<br>• Redes neuronales | Aumenta dimensionalidad con muchas categorías |
| **Binary Encoding** | Convierte cada valor a representación binaria | Categorías de alta cardinalidad | Menos interpretable |
| **Target Encoding** | Reemplaza categoría por la media de la variable objetivo | Variables categóricas predictivas con alta cardinalidad | Riesgo de sobreajuste |
| **Embedding** | Aprende representaciones vectoriales densas | Redes neuronales con muchas categorías | Requiere más datos y complejidad |

#### **Ejemplo práctico de implementación:**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

# Datos de ejemplo
data = {
    'color': ['rojo', 'azul', 'verde', 'rojo', 'verde'],
    'talla': ['S', 'M', 'L', 'XL', 'M'],
    'precio': [100, 120, 150, 180, 130]
}
df = pd.DataFrame(data)

# 1. Label Encoding - para algoritmos basados en árboles
label_encoder = LabelEncoder()
df['color_label'] = label_encoder.fit_transform(df['color'])
print("Label Encoding:\n", df[['color', 'color_label']])

# 2. One-Hot Encoding - para la mayoría de algoritmos
# Usando pandas (método simple)
color_dummies = pd.get_dummies(df['color'], prefix='color')
df_onehot = pd.concat([df, color_dummies], axis=1)
print("\nOne-Hot Encoding:\n", df_onehot)

# 3. Binary Encoding - eficiente para alta cardinalidad
binary_encoder = ce.BinaryEncoder(cols=['talla'])
df_binary = binary_encoder.fit_transform(df)
print("\nBinary Encoding:\n", df_binary)

# 4. Target Encoding - útil para categorías predictivas
target_encoder = ce.TargetEncoder(cols=['color'])
df_target = target_encoder.fit_transform(df['color'], df['precio'])
print("\nTarget Encoding:\n", df_target)
```

#### **Consideraciones avanzadas:**

- **Alta cardinalidad:** Para variables con muchas categorías (>50), considerar:
  - Agrupación de categorías poco frecuentes
  - Encoders jerárquicos
  - Técnicas de hashing

- **Nuevas categorías:** En producción pueden aparecer categorías nunca vistas:
  - Usar `handle_unknown='ignore'` en OneHotEncoder
  - Implementar estrategias de fallback

> **Ejercicio práctico:** Toma un dataset con variables categóricas y compara el rendimiento de diferentes técnicas de codificación con el mismo algoritmo.

---

### **3.3. Implementar selección de características estratégica**

**Problema:** Demasiadas características pueden causar sobreajuste, aumentar tiempo de entrenamiento y reducir la interpretabilidad del modelo.

**Solución:** Aplicar métodos de selección de características para identificar y mantener solo las variables más informativas.

#### **Métodos principales de selección:**

![Métodos de selección de características](https://i.imgur.com/placeholder_feature_selection.png)

| Método | Descripción | Ventajas | Limitaciones |
|--------|-------------|----------|--------------|
| **Filtro** | Evalúa características independientemente del modelo | • Rápido<br>• Simple<br>• Escalable | No considera interacciones entre variables |
| **Wrapper** | Evalúa subconjuntos usando el modelo | • Considera interacciones<br>• Específico para cada algoritmo | Computacionalmente costoso |
| **Embebido** | La selección ocurre durante el entrenamiento | • Balance entre filtro y wrapper<br>• Eficiente | Específico para ciertos algoritmos |

#### **Técnicas específicas con ejemplos de implementación:**

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Cargar dataset de ejemplo (dígitos escritos a mano)
X, y = load_digits(return_X_y=True)
print(f"Dimensiones originales: {X.shape}")

# 1. MÉTODO DE FILTRO: Selección univariada (ANOVA F-value)
selector_filtro = SelectKBest(f_classif, k=25)  # Seleccionar 25 mejores características
X_filtro = selector_filtro.fit_transform(X, y)
print(f"Después de filtro: {X_filtro.shape}")

# Evaluar impacto en rendimiento
pipeline_filtro = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=25)),
    ('classifier', SVC())
])
score_filtro = cross_val_score(pipeline_filtro, X, y, cv=5).mean()
print(f"Puntuación con filtro: {score_filtro:.4f}")

# 2. MÉTODO WRAPPER: Eliminación recursiva (RFE)
estimator = RandomForestClassifier(n_estimators=100)
selector_wrapper = RFE(estimator, n_features_to_select=25, step=1)
X_wrapper = selector_wrapper.fit_transform(X, y)
print(f"Después de wrapper: {X_wrapper.shape}")

# 3. MÉTODO EMBEBIDO: LASSO (L1 regularization)
lasso = Lasso(alpha=0.01)
X_scaled = StandardScaler().fit_transform(X)
lasso.fit(X_scaled, y)

# Visualizar importancia de características
importancia = np.abs(lasso.coef_)
indices = np.argsort(importancia)[::-1]
print("Top 10 características (LASSO):")
for i in range(10):
    print(f"Característica {indices[i]}: {importancia[indices[i]]:.4f}")
```

#### **Comparando resultados con y sin selección:**

La aplicación de técnicas de selección de características debe evaluarse con validación cruzada. Ejemplo comparativo:

```python
from sklearn.model_selection import GridSearchCV

# Pipeline sin selección de características
pipeline_completo = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Pipeline con selección
pipeline_seleccion = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif)),
    ('classifier', SVC())
])

# Buscar mejor número de características
param_grid = {
    'selector__k': [10, 20, 30, 40, 50, 64]  # 64 = todas las características
}

grid = GridSearchCV(pipeline_seleccion, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)

print(f"Mejor k: {grid.best_params_['selector__k']}")
print(f"Mejor puntuación: {grid.best_score_:.4f}")
print(f"Puntuación sin selección: {cross_val_score(pipeline_completo, X, y, cv=5).mean():.4f}")
```

> **Consejo práctico:** Inicia siempre con análisis de correlación y visualizaciones para entender las relaciones entre características antes de aplicar métodos automatizados.

---

### **3.4. Aplicar reducción de dimensionalidad cuando sea beneficioso**

**Problema:** Datasets con muchas dimensiones sufren de la "maldición de la dimensionalidad", donde la distancia entre puntos se vuelve menos significativa y el rendimiento se deteriora.

**Solución:** Aplicar técnicas de reducción de dimensionalidad para transformar el espacio de características manteniendo la información más relevante.

#### **Diferencia con selección de características:**

La **selección de características** conserva un subconjunto de las variables originales, mientras que la **reducción de dimensionalidad** crea nuevas variables que son combinaciones de las originales.

#### **Principales técnicas y sus aplicaciones:**

| Técnica | Tipo | Mejor para | Consideraciones |
|---------|------|------------|-----------------|
| **PCA** (Análisis de Componentes Principales) | Lineal | • Datos con correlaciones lineales<br>• Visualización<br>• Eliminación de ruido | • Sensible a escala<br>• No preserva distancias entre clases |
| **t-SNE** | No lineal | • Visualización<br>• Detección de clusters<br>• Datos complejos | • Computacionalmente intensivo<br>• No adecuado para proyección de nuevos datos |
| **UMAP** | No lineal | • Alternativa más rápida a t-SNE<br>• Conservación de estructura local y global | • Más reciente, menos establecido |
| **Autoencoder** | No lineal | • Datos muy complejos (imágenes, audio)<br>• Capturar relaciones no lineales | • Requiere más datos<br>• Más difícil de implementar y ajustar |

#### **Implementación de PCA en Python:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

# Cargar y preparar datos de ejemplo
wine = load_wine()
X = wine.data
y = wine.target
features = wine.feature_names

# Escalar datos (crucial para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Analizar varianza explicada
varianza_explicada = pca.explained_variance_ratio_
varianza_acumulada = np.cumsum(varianza_explicada)

# Visualizar resultados
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(varianza_explicada) + 1), varianza_explicada, alpha=0.5, label='Varianza individual')
plt.step(range(1, len(varianza_acumulada) + 1), varianza_acumulada, where='mid', label='Varianza acumulada')
plt.axhline(y=0.95, color='r', linestyle='--', label='Umbral 95%')
plt.xlabel('Número de componentes')
plt.ylabel('Ratio de varianza explicada')
plt.legend()
plt.title('Análisis de componentes principales (PCA)')

# Encontrar número óptimo de componentes (95% de varianza)
n_components = np.argmax(varianza_acumulada >= 0.95) + 1
print(f"Número de componentes para explicar 95% de varianza: {n_components}")

# Aplicar PCA con número óptimo de componentes
pca_optimo = PCA(n_components=n_components)
X_reducido = pca_optimo.fit_transform(X_scaled)
print(f"Dimensiones reducidas: {X_reducido.shape}")

# Visualizar primeros dos componentes
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.8, s=50)
plt.colorbar(scatter, label='Clase de vino')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Visualización de PCA: primeros dos componentes')

# Analizar contribución de variables originales
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_df = pd.DataFrame(loadings[:, :2], columns=['PC1', 'PC2'], index=features)
print("\nContribución de variables originales a PC1 y PC2:")
print(loading_df.sort_values(by='PC1', ascending=False))
```

#### **Comparando rendimiento antes y después:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Pipeline sin reducción
pipeline_completo = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# Pipeline con PCA
pipeline_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=n_components)),
    ('svm', SVC())
])

# Comparar rendimiento
score_completo = cross_val_score(pipeline_completo, X, y, cv=5).mean()
score_pca = cross_val_score(pipeline_pca, X, y, cv=5).mean()

print(f"Puntuación sin PCA: {score_completo:.4f}")
print(f"Puntuación con PCA ({n_components} componentes): {score_pca:.4f}")
print(f"Mejora: {(score_pca - score_completo) * 100:.2f}%")
```

> **Consejo práctico:** Para datos de alta dimensionalidad, prueba diferentes técnicas de reducción (PCA, t-SNE, UMAP) y compara los resultados visuales para entender la estructura de tus datos antes de decidir qué técnica usar.

---

### **3.5. Escalar características adecuadamente según el algoritmo**

**Problema:** Muchos algoritmos de ML son sensibles a la escala de las variables, lo que puede introducir sesgos o ralentizar la convergencia cuando las características tienen magnitudes muy diferentes.

**Solución:** Aplicar técnicas de transformación de escala apropiadas según el algoritmo y la distribución de los datos.

#### **¿Por qué escalar es importante?**

Imagina dos características: "ingresos_anuales" (rango: 20,000-200,000) y "edad" (rango: 18-90). Sin escalar, el algoritmo dará mucho más peso a "ingresos_anuales" simplemente porque sus valores son más grandes.

#### **Técnicas de escalado y normalización:**

| Técnica | Transformación | Mejor para | Uso recomendado |
|---------|----------------|------------|-----------------|
| **StandardScaler** | μ=0, σ=1 | Datos con distribución aproximadamente normal | • Regresión lineal<br>• SVM<br>• PCA<br>• Clustering |
| **MinMaxScaler** | [0,1] o [-1,1] | Datos con distribución desconocida o no normal | • KNN<br>• Redes neuronales<br>• Algoritmos con regularización L1/L2 |
| **RobustScaler** | Basado en cuartiles | Datos con valores atípicos (outliers) | • Regresión robusta<br>• Cuando hay outliers significativos |
| **Normalizer** | Norma L1/L2 = 1 | Vectores (no escalares) | • Vectores de texto<br>• Cuando solo importa la dirección |
| **QuantileTransformer** | Distribución normal o uniforme | Cualquier distribución | • Distribuciones muy sesgadas<br>• Datos no lineales |

#### **Implementación y comparación de métodos:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.datasets import load_boston

# Cargar dataset y seleccionar dos características con escalas diferentes
boston = load_boston()
X = boston.data
features = boston.feature_names
df = pd.DataFrame(X, columns=features)

# Seleccionar dos características con escalas distintas (ejemplo: RM y LSTAT)
X_ejemplo = df[['RM', 'LSTAT']].values
feature_names = ['RM (habitaciones)', 'LSTAT (% estatus bajo)']

# Técnicas de escalado
scalers = {
    'Datos originales': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer (normal)': QuantileTransformer(output_distribution='normal')
}

# Visualizar efectos
plt.figure(figsize=(15, 12))
i = 1

for name, scaler in scalers.items():
    plt.subplot(3, 2, i)
    
    if scaler:
        X_scaled = scaler.fit_transform(X_ejemplo)
    else:
        X_scaled = X_ejemplo.copy()
    
    # Scatter plot
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'{name}')
    
    # Añadir estadísticas
    if i > 1:  # No para datos originales
        for j, feature in enumerate(feature_names):
            plt.annotate(f'Media: {X_scaled[:, j].mean():.2f}, Std: {X_scaled[:, j].std():.2f}',
                        xy=(0.05, 0.95 - j*0.05), xycoords='axes fraction')
    
    plt.grid(True, alpha=0.3)
    i += 1

plt.tight_layout()
plt.show()
```

#### **Algoritmos y su necesidad de escalado:**

| Algoritmo | ¿Necesita escalado? | ¿Por qué? |
|-----------|---------------------|-----------|
| **Regresión Lineal/Logística** | **Sí** | • Coeficientes comparables<br>• Convergencia más rápida con SGD |
| **SVM** | **Sí** | • Se basa en distancias<br>• Muy sensible a escalas |
| **K-Means** | **Sí** | • Usa distancias euclidianas |
| **KNN** | **Sí** | • Basado completamente en distancias |
| **Árboles (Decision Tree, Random Forest)** | **No** | • Usan reglas de partición, no distancias |
| **Naïve Bayes** | **No** | • Basado en probabilidades |
| **Redes Neuronales** | **Sí** | • Convergencia más rápida<br>• Evita saturación de neuronas |

#### **Consideraciones importantes:**

- **Aplicar el escalado después de la división train/test** para evitar data leakage
- **Guardar los parámetros del scaler** para aplicar la misma transformación a datos nuevos
- **Escalar según el algoritmo**, no según el dataset
- **Combinar con imputación** para manejar valores faltantes antes de escalar

```python
# Implementación correcta con Pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, boston.target, test_size=0.2, random_state=42)

# Pipeline con escalado DENTRO del proceso
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # El scaler se entrena SOLO con datos de entrenamiento
    ('svr', SVR())
])

# Entrenamiento
pipeline.fit(X_train, y_train)

# Predicción (el escalado se aplica automáticamente)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Error cuadrático medio: {mse:.4f}")
```

> **Pregunta para reflexionar:** ¿Qué ocurriría si aplicaras MinMaxScaler a un conjunto de datos con outliers extremos? ¿Cómo afectaría esto a tu modelo?



---

### **3.6. Realizar ingeniería de características con conocimiento del dominio**

**Problema:** Los modelos genéricos no capturan completamente las relaciones específicas del dominio, limitando su capacidad predictiva.

**Solución:** Aprovechar el conocimiento del negocio y del dominio para diseñar características que incorporen la experiencia humana y la intuición del sector.

#### **Beneficios de la ingeniería de características basada en dominio:**

| Beneficio | Descripción |
|-----------|-------------|
| **Mayor poder predictivo** | Las características específicas del dominio suelen tener mayor correlación con la variable objetivo |
| **Modelos más interpretables** | Las características derivadas tienen significado para los expertos del dominio |
| **Menor necesidad de datos** | El conocimiento humano puede compensar parcialmente la escasez de datos |
| **Mejor generalización** | Capturan relaciones causales en lugar de correlaciones espurias |

#### **Ejemplos por industria:**

**1. Finanzas e inversión:**
```python
import pandas as pd
import numpy as np

# Asumiendo un DataFrame con datos de precios de acciones
df = pd.DataFrame({
    'fecha': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'precio_apertura': np.random.normal(100, 5, 100),
    'precio_cierre': np.random.normal(101, 5, 100),
    'precio_maximo': np.random.normal(103, 3, 100),
    'precio_minimo': np.random.normal(98, 3, 100),
    'volumen': np.random.normal(1000000, 200000, 100)
})

# Características inspiradas en análisis técnico
df['rango_diario'] = df['precio_maximo'] - df['precio_minimo']
df['retorno_diario'] = (df['precio_cierre'] - df['precio_apertura']) / df['precio_apertura']
df['volumen_relativo'] = df['volumen'] / df['volumen'].rolling(window=10).mean()

# Indicadores técnicos comunes
df['media_movil_10d'] = df['precio_cierre'].rolling(window=10).mean()
df['RSI_14'] = calcular_rsi(df['precio_cierre'], 14)  # Función personalizada para RSI

# Patrones de velas japonesas (simplificado)
df['doji'] = np.abs(df['precio_cierre'] - df['precio_apertura']) < 0.1 * df['rango_diario']
```

**2. Marketing y análisis de clientes:**
```python
# Extracción de componentes temporales
df['hora_dia'] = df['timestamp'].dt.hour
df['dia_semana'] = df['timestamp'].dt.day_name()
df['fin_semana'] = df['timestamp'].dt.dayofweek >= 5
df['mes'] = df['timestamp'].dt.month
df['trimestre'] = df['timestamp'].dt.quarter

# Categorización por tiempo (conocimiento de dominio)
df['periodo_dia'] = pd.cut(
    df['hora_dia'],
    bins=[0, 6, 12, 18, 24],
    labels=['madrugada', 'mañana', 'tarde', 'noche']
)

# Agregaciones temporales (comportamiento histórico)
clientes = df.groupby('cliente_id').agg({
    'compra': [
        ('total_compras', 'count'),
        ('compras_ultimo_mes', lambda x: x.iloc[-30:].sum() if len(x) >= 30 else x.sum()),
        ('dias_desde_ultima_compra', lambda x: (pd.Timestamp.now() - x.index[-1]).days)
    ],
    'monto': [
        ('monto_promedio', 'mean'),
        ('monto_total', 'sum'),
        ('monto_max', 'max')
    ]
})
```

**3. Medicina y salud:**
```python
# Características médicas derivadas
pacientes['imc'] = pacientes['peso'] / (pacientes['altura'] ** 2)
pacientes['relacion_cintura_cadera'] = pacientes['medida_cintura'] / pacientes['medida_cadera']

# Categorización según directrices médicas (conocimiento del dominio)
pacientes['categoria_imc'] = pd.cut(
    pacientes['imc'],
    bins=[0, 18.5, 25, 30, 100],
    labels=['bajo_peso', 'normal', 'sobrepeso', 'obesidad']
)

# Combinar factores de riesgo (conocimiento médico)
factores_riesgo = ['hipertension', 'diabetes', 'tabaquismo', 'colesterol_alto']
pacientes['num_factores_riesgo'] = pacientes[factores_riesgo].sum(axis=1)
pacientes['alto_riesgo_cv'] = (pacientes['edad'] > 50) & (pacientes['num_factores_riesgo'] >= 2)
```

#### **Enfoques generales aplicables a múltiples dominios:**

1. **Extraer componentes temporales significativos:**
   - Hora del día, día de la semana, mes, temporada
   - Proximidad a eventos especiales o feriados
   - Categorizar en periodos lógicos (mañana/tarde/noche)

2. **Crear agregaciones con ventanas temporales:**
   - Medias móviles (7 días, 30 días, 90 días)
   - Valores mínimos/máximos en periodos relevantes
   - Tasas de cambio entre periodos

3. **Distancias y relaciones espaciales:**
   - Distancia a puntos de interés
   - Densidad de población o características en un radio
   - Relaciones topológicas (dentro, adyacente, etc.)

> **Ejercicio de aplicación:** Piensa en un proyecto de ML y enumera 5 características específicas del dominio que un experto del sector consideraría importantes pero que un algoritmo automático probablemente no detectaría.

---

### **3.7. Realizar ingeniería de características sin conocimiento del dominio**

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

### **3.8. Documentar rigurosamente la ingeniería de características**

**Problema:** Con el tiempo, los equipos de ciencia de datos olvidan cómo se crearon las características, lo que dificulta la depuración, el mantenimiento y el conocimiento institucional cuando hay rotación de personal.

**Solución:** Implementar un sistema de documentación estructurado que registre todo el proceso de ingeniería de características.

#### **Componentes esenciales de la documentación:**

| Componente | Descripción | Ejemplo |
|------------|-------------|---------|
| **Nombre y descripción** | Nombre claro y explicación del significado | `dias_desde_ultima_compra`: Días transcurridos desde la última transacción del cliente |
| **Fórmula o algoritmo** | Cómo se calcula exactamente | `(fecha_actual - max(fechas_compra))` |
| **Justificación** | Por qué se creó y qué predice | Captura la recencia de actividad, predictor clave en modelos RFM |
| **Fuentes de datos** | Tablas y campos originales utilizados | Tabla `transacciones.cliente_id`, `transacciones.fecha` |
| **Transformaciones** | Procesos aplicados | Agrupación por cliente, extracción de máximo, diferencia de fechas |
| **Restricciones o limitaciones** | Casos donde puede fallar o ser inválida | Clientes nuevos tendrán valor nulo |
| **Autor y fecha** | Quién la creó y cuándo | Ana Martínez, 2023-04-15 |

#### **Sistema de documentación práctico:**

```python
import pandas as pd
from dataclasses import dataclass
import json
from datetime import datetime
import inspect

@dataclass
class FeatureDocumentation:
    """Clase para documentar características creadas"""
    nombre: str
    descripcion: str
    formula: str
    justificacion: str
    fuentes_datos: list
    transformaciones: list
    restricciones: list = None
    autor: str = None
    fecha_creacion: str = None
    codigo_fuente: str = None
    
    def __post_init__(self):
        if self.autor is None:
            self.autor = "Sistema automático"
        if self.fecha_creacion is None:
            self.fecha_creacion = datetime.now().strftime("%Y-%m-%d")
        if self.restricciones is None:
            self.restricciones = []
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}
    
    def to_json(self, filepath=None):
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return f"Documentación guardada en {filepath}"
        else:
            return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def to_markdown(self):
        md = f"# Característica: {self.nombre}\n\n"
        md += f"**Descripción:** {self.descripcion}\n\n"
        md += f"**Fórmula:** `{self.formula}`\n\n"
        md += f"**Justificación:** {self.justificacion}\n\n"
        md += "**Fuentes de datos:**\n"
        for fuente in self.fuentes_datos:
            md += f"- {fuente}\n"
        md += "\n**Transformaciones aplicadas:**\n"
        for trans in self.transformaciones:
            md += f"- {trans}\n"
        if self.restricciones:
            md += "\n**Restricciones o limitaciones:**\n"
            for res in self.restricciones:
                md += f"- {res}\n"
        md += f"\n**Autor:** {self.autor} | **Fecha:** {self.fecha_creacion}\n"
        if self.codigo_fuente:
            md += f"\n**Código fuente:**\n```python\n{self.codigo_fuente}\n```\n"
        return md


# Ejemplo de uso para documentar una característica
def crear_dias_desde_ultima_compra(df_transacciones):
    """Calcula los días transcurridos desde la última compra por cliente"""
    # Agrupar por cliente y obtener la fecha más reciente
    ultimas_compras = df_transacciones.groupby('cliente_id')['fecha'].max()
    
    # Calcular días desde esa fecha
    hoy = pd.Timestamp.now().normalize()
    dias_desde_ultima = (hoy - ultimas_compras).dt.days
    
    # Crear un DataFrame con el resultado
    resultado = pd.DataFrame({
        'cliente_id': dias_desde_ultima.index,
        'dias_desde_ultima_compra': dias_desde_ultima.values
    })
    
    # Documentar la característica
    doc = FeatureDocumentation(
        nombre="dias_desde_ultima_compra",
        descripcion="Número de días transcurridos desde la última transacción del cliente",
        formula="(fecha_actual - max(fechas_compra_cliente))",
        justificacion="Indicador de recencia que ayuda a predecir probabilidad de abandono y valor del cliente",
        fuentes_datos=["transacciones.cliente_id", "transacciones.fecha"],
        transformaciones=["Agrupación por cliente_id", "Extracción de fecha máxima", "Cálculo de diferencia con fecha actual"],
        restricciones=["Clientes sin compras tendrán valores nulos", "Sensible a la zona horaria del sistema"],
        autor="Equipo de Ciencia de Datos",
        codigo_fuente=inspect.getsource(crear_dias_desde_ultima_compra)
    )
    
    # Guardar documentación
    doc.to_json(f"docs/features/dias_desde_ultima_compra_{doc.fecha_creacion}.json")
    
    return resultado, doc

# Crear la característica (con datos ficticios para el ejemplo)
# df_resultado, documentacion = crear_dias_desde_ultima_compra(df_transacciones)
```

#### **Repositorio centralizado de características:**

Un sistema más avanzado incluiría:

1. **Catálogo de características** accesible para todo el equipo
2. **Control de versiones** para las definiciones de características
3. **Linaje de datos** que rastrea el origen y las transformaciones
4. **Métricas de uso** que muestren qué modelos utilizan cada característica
5. **Sistema de búsqueda** para encontrar características existentes

> **Pregunta de reflexión:** ¿Cuántas veces has tenido que recrear una característica porque no recordabas exactamente cómo se construyó originalmente? ¿Qué problemas te habría evitado una documentación adecuada?

---

### **3.9. Dominar técnicas de extracción de características de texto**

**Problema:** Los datos de texto son no estructurados por naturaleza y requieren transformaciones especiales para ser utilizados en modelos de ML convencionales.

**Solución:** Aplicar técnicas de procesamiento de lenguaje natural (NLP) para transformar texto en representaciones numéricas que capturen la semántica y el contexto.

#### **Flujo de trabajo para procesamiento de texto:**

![Flujo de trabajo NLP](https://i.imgur.com/placeholder_nlp_workflow.png)

#### **1. Preprocesamiento de texto**

Antes de cualquier extracción de características, el texto debe limpiarse y normalizarse:

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Descargar recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocesar_texto(texto, opciones=None):
    """
    Preprocesa un texto aplicando varias técnicas según las opciones seleccionadas.
    
    Parámetros:
        texto (str): El texto a preprocesar
        opciones (dict): Diccionario con opciones de preprocesamiento
    
    Retorna:
        str: Texto preprocesado
    """
    if opciones is None:
        opciones = {
            'minusculas': True,
            'eliminar_urls': True,
            'eliminar_html': True,
            'eliminar_puntuacion': True,
            'eliminar_numeros': True,
            'eliminar_stopwords': True,
            'stemming': False,
            'lemmatization': True,
            'idioma': 'spanish'
        }
    
    # Convertir a minúsculas
    if opciones.get('minusculas', True):
        texto = texto.lower()
    
    # Eliminar URLs
    if opciones.get('eliminar_urls', True):
        texto = re.sub(r'https?://\S+|www\.\S+', '', texto)
    
    # Eliminar etiquetas HTML
    if opciones.get('eliminar_html', True):
        texto = re.sub(r'<.*?>', '', texto)
    
    # Eliminar puntuación
    if opciones.get('eliminar_puntuacion', True):
        texto = re.sub(r'[^\w\s]', '', texto)
    
    # Eliminar números
    if opciones.get('eliminar_numeros', True):
        texto = re.sub(r'\d+', '', texto)
    
    # Tokenización
    tokens = nltk.word_tokenize(texto)
    
    # Eliminar stopwords
    if opciones.get('eliminar_stopwords', True):
        stop_words = set(stopwords.words(opciones.get('idioma', 'spanish')))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming (reducción a raíz)
    if opciones.get('stemming', False):
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Lematización (reducción a forma base)
    if opciones.get('lemmatization', True):
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Reunir tokens
    texto_procesado = ' '.join(tokens)
    
    return texto_procesado

# Ejemplo de uso
texto_original = "¡Hola mundo! Este es un ejemplo de texto con URLs como https://example.com y algunas palabras repetidas repetidas."
texto_procesado = preprocesar_texto(texto_original)
print(f"Original: {texto_original}")
print(f"Procesado: {texto_procesado}")
```

#### **2. Enfoques de vectorización de texto**

| Técnica | Descripción | Ventajas | Limitaciones | Mejor para |
|---------|-------------|----------|-------------|------------|
| **Bag of Words (BoW)** | Cuenta de palabras sin orden | Simple, intuitivo | Pierde orden y contexto | Clasificación básica, análisis exploratorio |
| **TF-IDF** | Frecuencia de término × Inversa de frecuencia en documentos | Resalta términos importantes | Sigue siendo disperso, sin semántica | Clasificación, búsqueda, sistemas de recomendación |
| **Word Embeddings** | Vectores densos que representan significado | Captura relaciones semánticas | Requiere más datos | NLP avanzado, procesamiento semántico |
| **Transformadores (BERT, etc.)** | Modelos contextuales profundos | Captura contexto bidireccional | Computacionalmente intensivos | Tareas complejas de comprensión del lenguaje |

##### **2.1 TF-IDF en la práctica:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Ejemplo de corpus
corpus = [
    "Este es el primer documento con algunas palabras.",
    "Este documento es el segundo documento.",
    "Y este es el tercer documento con más palabras.",
    "¿Es este el primer documento del corpus?"
]

# Crear vectorizador TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()

# Convertir a DataFrame para mejor visualización
df_tfidf = pd.DataFrame(X.toarray(), columns=terms)
print("Matriz TF-IDF:")
print(df_tfidf.round(2))

# Identificar palabras más importantes en cada documento
for i, doc in enumerate(corpus):
    print(f"\nPalabras más importantes en documento {i+1}:")
    # Obtener las 3 palabras con mayor puntuación TF-IDF
    tfidf_sorting = np.argsort(X[i].toarray()[0])[::-1]
    top_n = 3
    for idx in tfidf_sorting[:top_n]:
        if X[i, idx] > 0:
            print(f"  - {terms[idx]}: {X[i, idx]:.4f}")
```

##### **2.2 Word Embeddings con Gensim:**

```python
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Preparar frases tokenizadas
sentences = [
    ["este", "es", "el", "primer", "documento", "con", "algunas", "palabras"],
    ["este", "documento", "es", "el", "segundo", "documento"],
    ["y", "este", "es", "el", "tercer", "documento", "con", "más", "palabras"],
    ["es", "este", "el", "primer", "documento", "del", "corpus"]
]

# Entrenar modelo Word2Vec
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Visualizar embeddings en 2D
def plot_embeddings(model, words):
    X = model.wv[words]
    
    # Reducir dimensiones para visualización
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    # Crear gráfico
    plt.figure(figsize=(10, 7))
    plt.scatter(result[:, 0], result[:, 1], c='steelblue', s=100, alpha=0.7)
    
    # Añadir etiquetas
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]), fontsize=12)
    
    plt.title("Proyección 2D de Word Embeddings", fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Visualizar palabras seleccionadas
palabras_interes = ["documento", "primer", "segundo", "tercer", "palabras", "corpus"]
plot_embeddings(model, palabras_interes)

# Explorar relaciones semánticas
print("Palabras más similares a 'documento':")
similares = model.wv.most_similar("documento", topn=5)
for word, score in similares:
    print(f"  - {word}: {score:.4f}")

# Analogías vectoriales (cuando hay suficientes datos)
# resultado = model.wv.most_similar(positive=['mujer', 'rey'], negative=['hombre'])
# print("rey - hombre + mujer =", resultado[0][0])
```

#### **3. Técnicas avanzadas con transformadores**

Para tareas más sofisticadas, los modelos de transformadores como BERT han revolucionado el NLP:

```python
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Cargar modelo y tokenizador preentrenados
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")

# Función para obtener embeddings contextuales
def get_bert_embedding(text, pooling='mean'):
    """Obtiene embedding contextual usando BERT"""
    # Add special tokens
    marked_text = "[CLS] " + text + " [SEP]"
    
    # Tokenizar
    tokenized_text = tokenizer.tokenize(marked_text)
    
    # Mapear tokens a IDs
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    
    # Crear tensor de segmentos (todos 0 para una sola frase)
    segments_ids = [0] * len(indexed_tokens)
    
    # Convertir a tensores
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    # Obtener embeddings (sin gradientes para inferencia)
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[0]
    
    # Diferentes estrategias de pooling
    if pooling == 'cls':
        # Usar el token [CLS] como representación de la frase
        sentence_embedding = hidden_states[0][0]
    else:  # mean pooling
        # Promediar todos los tokens (excluyendo [CLS] y [SEP])
        token_embeddings = hidden_states[0]
        sentence_embedding = torch.mean(token_embeddings[1:-1], dim=0)
    
    return sentence_embedding.numpy()

# Ejemplo de uso
frases = [
    "Me encantó esta película, el argumento fue excelente.",
    "No recomendaría este restaurante, la comida estaba fría.",
    "El servicio al cliente fue impecable y rápido."
]

# Obtener embeddings
embeddings = [get_bert_embedding(frase) for frase in frases]

# Calcular similitud coseno entre embeddings
from sklearn.metrics.pairwise import cosine_similarity
sim_matrix = cosine_similarity(embeddings)

print("Matriz de similitud coseno:")
for i in range(len(frases)):
    for j in range(len(frases)):
        print(f"Similitud entre frase {i+1} y frase {j+1}: {sim_matrix[i][j]:.4f}")
```

#### **4. Recomendaciones para elegir el enfoque adecuado**

| Escenario | Enfoque recomendado |
|-----------|---------------------|
| **Clasificación simple con pocos datos** | Bag of Words o TF-IDF |
| **Clasificación con datos de tamaño medio** | Word Embeddings preentrenados |
| **Comprensión de lenguaje complejo** | Transformers (BERT, RoBERTa, etc.) |
| **Limitaciones computacionales** | N-gramas con selección de características |
| **Necesidad de explicabilidad** | TF-IDF con análisis de coeficientes |
| **Multilingüe o dominio específico** | Embeddings especializados o fine-tuning de transformers |

> **Ejercicio práctico:** Toma un conjunto de textos (reseñas, tweets, artículos, etc.) y compara el rendimiento de un modelo de clasificación usando diferentes técnicas de extracción de características (Bag of Words, TF-IDF, Word Embeddings y BERT). ¿Qué técnica funciona mejor y por qué?

## **4. Mejores prácticas en la etapa de entrenamiento, evaluación y selección del modelo**

> *"Todo modelo está equivocado, pero algunos son útiles."* - George Box

El proceso de entrenamiento, evaluación y selección de modelos es donde finalmente convertimos los datos preparados en un sistema predictivo. Aunque es tentador enfocarse inmediatamente en la precisión, es crucial adoptar un enfoque sistemático que equilibre múltiples factores: rendimiento, interpretabilidad, velocidad y mantenibilidad.

![Ciclo de entrenamiento y evaluación](https://i.imgur.com/placeholder_model_selection.png)

### **4.1. Seleccionar algoritmos iniciales estratégicamente**

**Problema:** Con tantos algoritmos disponibles, es imposible evaluarlos todos exhaustivamente, mientras que elegir uno al azar puede llevar a resultados subóptimos.

**Solución:** Seleccionar estratégicamente 2-3 algoritmos iniciales basados en las características del problema, los datos disponibles y los requisitos del proyecto.

#### **Marco para la selección de algoritmos:**

| Criterio | Preguntas clave | Impacto en la selección |
|----------|----------------|------------------------|
| **Tipo de problema** | ¿Clasificación, regresión, clustering, recomendación, etc.? | Define la categoría de algoritmos aplicables |
| **Volumen de datos** | ¿Cuántos ejemplos de entrenamiento tenemos? | Algunos algoritmos necesitan más datos que otros |
| **Dimensionalidad** | ¿Cuántas características hay? | Algoritmos como KNN sufren con alta dimensionalidad |
| **Escalabilidad requerida** | ¿El modelo necesita procesar datos en tiempo real? | Impacta en la elección de algoritmos eficientes |
| **Interpretabilidad** | ¿Necesitamos explicar las predicciones? | Favorece árboles de decisión vs. redes neuronales |
| **Balance sesgo-varianza** | ¿Preferimos generalización o ajuste preciso? | Guía entre modelos simples y complejos |
| **Distribución de datos** | ¿Los datos son linealmente separables? | Indica si son necesarios métodos no lineales |
| **Restricciones técnicas** | ¿Limitaciones de memoria, cómputo o despliegue? | Elimina algoritmos no viables |

#### **Algoritmos recomendados por escenario:**

![Mapa de selección de algoritmos](https://i.imgur.com/placeholder_algorithm_map.png)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Función para recomendar algoritmos basados en características del problema
def recomendar_algoritmos(tipo_problema, num_muestras, num_caracteristicas, 
                          interpretabilidad_requerida=False, datos_lineales=None,
                          tiempo_real=False, desbalanceado=False):
    """
    Recomienda algoritmos iniciales para explorar basados en las características del problema.
    
    Parámetros:
    -----------
    tipo_problema : str
        'clasificacion' o 'regresion'
    num_muestras : int
        Número de ejemplos de entrenamiento
    num_caracteristicas : int
        Número de características/dimensiones
    interpretabilidad_requerida : bool
        Si se requiere que el modelo sea interpretable
    datos_lineales : bool o None
        Si los datos presentan relaciones lineales (None si se desconoce)
    tiempo_real : bool
        Si el modelo necesita hacer predicciones en tiempo real
    desbalanceado : bool
        Si las clases están muy desbalanceadas (solo para clasificación)
    
    Retorna:
    --------
    dict
        Algoritmos recomendados con prioridad y razones
    """
    recomendaciones = {}
    
    if tipo_problema == 'clasificacion':
        # Pocos datos (< 10k ejemplos)
        if num_muestras < 10000:
            recomendaciones['Naive Bayes'] = {
                'prioridad': 'Alta',
                'razones': ['Funciona bien con pocos datos', 'Rápido de entrenar'],
                'import': 'from sklearn.naive_bayes import GaussianNB, MultinomialNB'
            }
            
            recomendaciones['SVM'] = {
                'prioridad': 'Alta',
                'razones': ['Efectivo en espacios de alta dimensión', 'Buena generalización'],
                'import': 'from sklearn.svm import SVC'
            }
            
            if interpretabilidad_requerida:
                recomendaciones['Árbol de decisión'] = {
                    'prioridad': 'Alta',
                    'razones': ['Altamente interpretable', 'Puede visualizarse'],
                    'import': 'from sklearn.tree import DecisionTreeClassifier'
                }
        
        # Datos medianos a grandes (≥ 10k ejemplos)
        else:
            recomendaciones['Random Forest'] = {
                'prioridad': 'Alta',
                'razones': ['Robusto contra overfitting', 'Maneja bien muchas características'],
                'import': 'from sklearn.ensemble import RandomForestClassifier'
            }
            
            recomendaciones['Gradient Boosting'] = {
                'prioridad': 'Alta',
                'razones': ['Alto rendimiento', 'Maneja características mixtas'],
                'import': 'from sklearn.ensemble import GradientBoostingClassifier, from xgboost import XGBClassifier'
            }
            
            if not interpretabilidad_requerida and num_muestras > 50000:
                recomendaciones['Redes Neuronales'] = {
                    'prioridad': 'Media',
                    'razones': ['Captura relaciones complejas', 'Requiere muchos datos'],
                    'import': 'from sklearn.neural_network import MLPClassifier'
                }
        
        # Si hay restricciones de tiempo o se requiere aprendizaje online
        if tiempo_real:
            recomendaciones['SGD Classifier'] = {
                'prioridad': 'Alta' if tiempo_real else 'Media',
                'razones': ['Rápido', 'Soporta aprendizaje incremental'],
                'import': 'from sklearn.linear_model import SGDClassifier'
            }
            
        # Para datos desbalanceados
        if desbalanceado:
            recomendaciones['Balanced Random Forest'] = {
                'prioridad': 'Alta',
                'razones': ['Maneja clases desbalanceadas', 'Reduce overfitting'],
                'import': 'from imblearn.ensemble import BalancedRandomForestClassifier'
            }
    
    elif tipo_problema == 'regresion':
        # Si hay evidencia de relaciones lineales o es desconocido
        if datos_lineales or datos_lineales is None:
            recomendaciones['Regresión Lineal/Ridge'] = {
                'prioridad': 'Alta',
                'razones': ['Simple', 'Rápido', 'Interpretable'],
                'import': 'from sklearn.linear_model import LinearRegression, Ridge'
            }
        
        # Para cualquier tamaño de dataset
        recomendaciones['Random Forest Regressor'] = {
            'prioridad': 'Alta',
            'razones': ['Robusto', 'Maneja no-linealidad', 'Pocos hiperparámetros críticos'],
            'import': 'from sklearn.ensemble import RandomForestRegressor'
        }
        
        # Para datasets más grandes
        if num_muestras >= 10000:
            recomendaciones['Gradient Boosting Regressor'] = {
                'prioridad': 'Alta',
                'razones': ['Alto rendimiento', 'Buena generalización'],
                'import': 'from sklearn.ensemble import GradientBoostingRegressor, from xgboost import XGBRegressor'
            }
        
        # Alta dimensionalidad y posible no linealidad
        if num_caracteristicas > 50 and (datos_lineales is False or datos_lineales is None):
            recomendaciones['SVM Regressor'] = {
                'prioridad': 'Media',
                'razones': ['Maneja espacios de alta dimensión', 'Captura relaciones no lineales con kernels'],
                'import': 'from sklearn.svm import SVR'
            }
    
    return recomendaciones

# Ejemplo de uso
tipo_problema = 'clasificacion'
num_muestras = 5000
num_caracteristicas = 20
interpretabilidad = True

recomendaciones = recomendar_algoritmos(
    tipo_problema=tipo_problema,
    num_muestras=num_muestras,
    num_caracteristicas=num_caracteristicas,
    interpretabilidad_requerida=interpretabilidad,
    datos_lineales=None,  # Desconocido
    tiempo_real=False,
    desbalanceado=True
)

print(f"Recomendaciones para {tipo_problema} con {num_muestras} muestras y {num_caracteristicas} características:")
for algoritmo, info in recomendaciones.items():
    print(f"\n{algoritmo} (Prioridad: {info['prioridad']})")
    print("Razones:")
    for razon in info['razones']:
        print(f"  - {razon}")
    print(f"Import: {info['import']}")
```

#### **Comparación de características por algoritmo:**

| Algoritmo | Interpretabilidad | Velocidad entrenamiento | Velocidad predicción | Memoria requerida | Manejo de outliers | Escalabilidad |
|-----------|-------------------|-------------------------|----------------------|-------------------|---------------------|--------------|
| **Regresión lineal/logística** | Alta | Alta | Alta | Baja | Baja | Alta |
| **Naïve Bayes** | Media | Alta | Alta | Baja | Media | Alta |
| **Árboles de decisión** | Alta | Media | Alta | Baja | Alta | Media |
| **Random Forest** | Media | Media | Media | Media | Alta | Media |
| **Gradient Boosting** | Baja | Baja | Media | Media | Alta | Baja |
| **SVM** | Baja | Baja (grandes datasets) | Media | Media-Alta | Media | Baja |
| **K-Nearest Neighbors** | Media | Alta | Baja | Alta | Baja | Baja |
| **Redes neuronales** | Muy baja | Muy baja | Alta | Alta | Media | Media-Alta |

#### **Implementación práctica de evaluación inicial:**

```python
def evaluar_algoritmos_iniciales(X, y, tipo_problema='clasificacion', cv=5, random_state=42):
    """
    Evalúa rápidamente varios algoritmos con configuraciones por defecto
    
    Parámetros:
    -----------
    X : array-like
        Características
    y : array-like
        Variable objetivo
    tipo_problema : str
        'clasificacion' o 'regresion'
    cv : int
        Número de folds para validación cruzada
    
    Retorna:
    --------
    DataFrame
        Resultados comparativos de los algoritmos
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    resultados = []
    algoritmos = {}
    
    if tipo_problema == 'clasificacion':
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        
        algoritmos = {
            'Regresión Logística': LogisticRegression(max_iter=1000, random_state=random_state),
            'Naive Bayes': GaussianNB(),
            'Árbol de Decisión': DecisionTreeClassifier(random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
            'SVM': SVC(probability=True, random_state=random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Métricas para clasificación
        metrics = {
            'Accuracy': (accuracy_score, {}),
            'F1 Score': (f1_score, {'average': 'weighted'})
        }
    
    else:  # regresión
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        
        algoritmos = {
            'Regresión Lineal': LinearRegression(),
            'Ridge': Ridge(random_state=random_state),
            'Árbol de Decisión': DecisionTreeRegressor(random_state=random_state),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        
        # Métricas para regresión
        metrics = {
            'MSE': (mean_squared_error, {}),
            'R²': (r2_score, {})
        }
    
    # Evaluar cada algoritmo
    for nombre, algoritmo in algoritmos.items():
        try:
            # Entrenar modelo
            inicio = time.time()
            algoritmo.fit(X_train_scaled, y_train)
            tiempo_train = time.time() - inicio
            
            # Tiempo de predicción
            inicio = time.time()
            y_pred = algoritmo.predict(X_test_scaled)
            tiempo_pred = time.time() - inicio
            
            # Calcular métricas
            scores = {}
            for nombre_metrica, (metrica, params) in metrics.items():
                scores[nombre_metrica] = metrica(y_test, y_pred, **params)
            
            # Validación cruzada
            cv_scores = cross_val_score(
                algoritmo, X_train_scaled, y_train, cv=cv, 
                scoring='accuracy' if tipo_problema == 'clasificacion' else 'neg_mean_squared_error'
            )
            
            resultados.append({
                'Algoritmo': nombre,
                'CV Score': np.mean(cv_scores) if tipo_problema == 'clasificacion' else -np.mean(cv_scores),
                'CV Std': np.std(cv_scores) if tipo_problema == 'clasificacion' else np.std(-cv_scores),
                **scores,
                'Tiempo Train (s)': tiempo_train,
                'Tiempo Pred (s)': tiempo_pred
            })
            
        except Exception as e:
            print(f"Error con {nombre}: {str(e)}")
            continue
    
    # Ordenar por rendimiento (primera métrica)
    primera_metrica = list(metrics.keys())[0]
    if tipo_problema == 'clasificacion':
        resultados.sort(key=lambda x: x[primera_metrica], reverse=True)
    else:
        resultados.sort(key=lambda x: x[primera_metrica])
    
    return pd.DataFrame(resultados)

# Ejemplo de uso (con datos hipotéticos)
# resultados = evaluar_algoritmos_iniciales(X, y, tipo_problema='clasificacion')
# print(resultados)
```

#### **Consejos para la selección inicial:**

1. **No te comprometas demasiado pronto:** Prueba varios algoritmos con configuración por defecto antes de profundizar.
2. **Combina algoritmos simples y complejos:** Un modelo lineal simple puede sorprendentemente superar modelos complejos.
3. **Considera todo el ciclo de vida:** El algoritmo más preciso puede no ser viable en producción por costos computacionales.
4. **Piensa en interpretabilidad vs. rendimiento:** ¿Necesitas explicar las predicciones o solo que sean precisas?
5. **Valora la mantenibilidad:** Los algoritmos más exóticos pueden ser difíciles de mantener a largo plazo.

> **Ejercicio práctico:** Para un problema que te interese, selecciona tres algoritmos iniciales siguiendo la guía anterior. Implementa cada uno con configuraciones por defecto y compara sus resultados. ¿Los algoritmos seleccionados funcionaron como esperabas? ¿Hubo alguna sorpresa?

### **4.2. Implementar estrategias efectivas contra el sobreajuste**

**Problema:** Los modelos con alta capacidad tienden a memorizar los datos de entrenamiento (sobreajuste), lo que resulta en un pobre rendimiento en datos nuevos.

**Solución:** Aplicar múltiples técnicas de regularización y validación para asegurar que el modelo generalice bien a datos no vistos.

#### **El sobreajuste en contexto:**

El sobreajuste ocurre cuando un modelo se ajusta demasiado a las peculiaridades y ruido de los datos de entrenamiento. Esto se manifiesta como:
- Alto rendimiento en datos de entrenamiento
- Bajo rendimiento en datos de validación/prueba
- Alta varianza en las predicciones

![Ilustración de sobreajuste](https://i.imgur.com/placeholder_overfitting.png)

#### **Estrategias probadas para combatir el sobreajuste:**

| Estrategia | Descripción | Implementación | Mejores para |
|------------|-------------|----------------|--------------|
| **Validación cruzada** | Evaluar modelos en múltiples particiones de datos | `sklearn.model_selection.cross_val_score` | Todos los modelos |
| **Regularización L1/L2** | Penalizar coeficientes grandes | `sklearn.linear_model.Ridge`, `Lasso` | Modelos lineales |
| **Poda (pruning)** | Reducir complejidad de árboles | `ccp_alpha` en árboles de decisión | Árboles |
| **Early stopping** | Detener entrenamiento cuando la validación empeora | `early_stopping` en modelos iterativos | Algoritmos iterativos |
| **Dropout** | Desactivar neuronas aleatoriamente | `nn.Dropout()` en redes neuronales | Redes neuronales |
| **Aumento de datos** | Generar datos sintéticos de entrenamiento | Transformaciones, ruido, etc. | Imágenes, series temporales |
| **Ensamblado** | Combinar múltiples modelos | `VotingClassifier`, `StackingRegressor` | Cualquier modelo |

#### **Implementación de validación cruzada:**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def evaluar_con_validacion_cruzada(X, y, modelo, cv=5, scoring='accuracy'):
    """
    Evalúa un modelo con validación cruzada y muestra resultados detallados
    
    Parámetros:
    -----------
    X : array-like
        Características
    y : array-like
        Variable objetivo
    modelo : estimator
        Modelo de scikit-learn
    cv : int
        Número de folds para validación cruzada
    scoring : str
        Métrica de evaluación
    
    Retorna:
    --------
    dict
        Estadísticas de validación cruzada
    """
    # Ejecutar validación cruzada
    scores = cross_val_score(modelo, X, y, cv=cv, scoring=scoring)
    
    # Estadísticas
    stats = {
        'promedio': scores.mean(),
        'desviacion_estandar': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'rango': scores.max() - scores.min()
    }
    
    # Imprimir resultados
    print(f"Validación cruzada ({cv} folds) para {modelo.__class__.__name__}:")
    print(f"  Métrica: {scoring}")
    print(f"  Promedio: {stats['promedio']:.4f} ± {stats['desviacion_estandar']:.4f}")
    print(f"  Rango: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Visualizar resultados por fold
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, cv+1), scores, alpha=0.8, color='steelblue')
    plt.axhline(y=stats['promedio'], color='red', linestyle='-', label=f'Promedio: {stats["promedio"]:.4f}')
    plt.fill_between(
        range(1, cv+1), 
        stats['promedio'] - stats['desviacion_estandar'], 
        stats['promedio'] + stats['desviacion_estandar'], 
        alpha=0.2, color='red', label=f'Desviación estándar: {stats["desviacion_estandar"]:.4f}'
    )
    plt.xlabel('Fold')
    plt.ylabel(scoring)
    plt.title(f'Resultados por fold para {modelo.__class__.__name__}')
    plt.xticks(range(1, cv+1))
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return stats

# Ejemplo de uso (con datos hipotéticos):
# modelo = RandomForestClassifier(n_estimators=100, random_state=42)
# stats = evaluar_con_validacion_cruzada(X, y, modelo, cv=5, scoring='accuracy')
```

#### **Implementación de regularización L1/L2:**

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def comparar_regularizacion(X, y, alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5):
    """
    Compara diferentes técnicas de regularización (Ridge, Lasso, ElasticNet)
    y encuentra el mejor valor de alpha.
    
    Parámetros:
    -----------
    X : array-like
        Características
    y : array-like
        Variable objetivo
    alphas : list
        Valores de alpha (parámetro de regularización) a probar
    cv : int
        Número de folds para validación cruzada
    
    Retorna:
    --------
    dict
        Mejores modelos y sus parámetros
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Escalar características (importante para regularización)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar modelos y grid search
    modelos = {
        'Ridge (L2)': (Ridge(), {'alpha': alphas}),
        'Lasso (L1)': (Lasso(max_iter=10000), {'alpha': alphas}),
        'ElasticNet (L1+L2)': (
            ElasticNet(max_iter=10000), 
            {'alpha': alphas, 'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]}
        )
    }
    
    mejores_modelos = {}
    
    # Evaluar cada tipo de regularización
    for nombre, (modelo, param_grid) in modelos.items():
        print(f"\nBuscando mejores parámetros para {nombre}...")
        grid = GridSearchCV(
            estimator=modelo,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=0,
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)
        
        # Guardar mejor modelo
        mejores_modelos[nombre] = {
            'modelo': grid.best_estimator_,
            'parametros': grid.best_params_,
            'mse_validacion': -grid.best_score_,
            'mse_test': mean_squared_error(
                y_test, 
                grid.best_estimator_.predict(X_test_scaled)
            )
        }
        
        print(f"  Mejores parámetros: {grid.best_params_}")
        print(f"  MSE Validación: {-grid.best_score_:.4f}")
        print(f"  MSE Test: {mejores_modelos[nombre]['mse_test']:.4f}")
    
    # Comparar coeficientes (para ver efecto de regularización)
    plt.figure(figsize=(12, 6))
    
    coef_ridge = mejores_modelos['Ridge (L2)']['modelo'].coef_
    coef_lasso = mejores_modelos['Lasso (L1)']['modelo'].coef_
    coef_elastic = mejores_modelos['ElasticNet (L1+L2)']['modelo'].coef_
    
    # Ordenar por magnitud en coeficientes Ridge
    indices_ordenados = np.argsort(np.abs(coef_ridge))[::-1]
    nombres_x = [f"X{i}" for i in indices_ordenados]
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(coef_ridge)), coef_ridge[indices_ordenados], alpha=0.7, label='Ridge (L2)')
    plt.bar(range(len(coef_lasso)), coef_lasso[indices_ordenados], alpha=0.7, label='Lasso (L1)')
    plt.bar(range(len(coef_elastic)), coef_elastic[indices_ordenados], alpha=0.7, label='ElasticNet')
    plt.xlabel('Características (ordenadas por importancia)')
    plt.ylabel('Coeficientes')
    plt.title('Comparación de coeficientes')
    plt.legend()
    plt.xticks([])
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(coef_ridge)), np.abs(coef_ridge[indices_ordenados]), alpha=0.7, label='Ridge (L2)')
    plt.bar(range(len(coef_lasso)), np.abs(coef_lasso[indices_ordenados]), alpha=0.7, label='Lasso (L1)')
    plt.bar(range(len(coef_elastic)), np.abs(coef_elastic[indices_ordenados]), alpha=0.7, label='ElasticNet')
    plt.xlabel('Características (ordenadas por importancia)')
    plt.ylabel('Valor absoluto de coeficientes')
    plt.title('Comparación de magnitudes')
    plt.legend()
    plt.xticks([])
    
    plt.tight_layout()
    plt.show()
    
    return mejores_modelos

# Ejemplo de uso (con datos hipotéticos):
# mejores_modelos = comparar_regularizacion(X, y, alphas=[0.001, 0.01, 0.1, 1, 10, 100])
```

#### **Implementación de poda (pruning) en árboles:**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV

def podar_arbol_decision(X, y, max_depth_range=[2, 3, 5, 10, 15, 20, None], 
                         ccp_alpha_range=[0.0, 0.001, 0.01, 0.05, 0.1], 
                         random_state=42):
    """
    Evalúa diferentes niveles de poda para un árbol de decisión
    
    Parámetros:
    -----------
    X : array-like
        Características
    y : array-like
        Variable objetivo
    max_depth_range : list
        Valores de profundidad máxima a evaluar
    ccp_alpha_range : list
        Valores de alpha para Cost-Complexity Pruning
    
    Retorna:
    --------
    tuple
        (mejor_modelo, grid_search_results)
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    
    # Parámetros para Grid Search
    param_grid = {
        'max_depth': max_depth_range,
        'ccp_alpha': ccp_alpha_range
    }
    
    # Configurar modelo base
    base_tree = DecisionTreeClassifier(random_state=random_state)
    
    # Grid Search
    grid = GridSearchCV(
        base_tree, param_grid, cv=5, 
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    
    # Evaluar mejor modelo
    mejor_modelo = grid.best_estimator_
    accuracy_train = mejor_modelo.score(X_train, y_train)
    accuracy_test = mejor_modelo.score(X_test, y_test)
    
    print(f"Mejores parámetros: {grid.best_params_}")
    print(f"Accuracy train: {accuracy_train:.4f}")
    print(f"Accuracy test: {accuracy_test:.4f}")
    
    # Visualizar árbol podado vs sin podar
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    
    # Árbol sin podar (solo limitado por max_depth)
    tree_unpruned = DecisionTreeClassifier(
        max_depth=grid.best_params_['max_depth'],
        ccp_alpha=0.0,
        random_state=random_state
    )
    tree_unpruned.fit(X_train, y_train)
    
    # Árbol podado
    tree_pruned = DecisionTreeClassifier(
        max_depth=grid.best_params_['max_depth'],
        ccp_alpha=grid.best_params_['ccp_alpha'],
        random_state=random_state
    )
    tree_pruned.fit(X_train, y_train)
    
    # Visualizar árboles
    plot_tree(
        tree_unpruned, filled=True, feature_names=None, 
        class_names=None, ax=axes[0], max_depth=3
    )
    axes[0].set_title(f"Árbol sin poda (max_depth={grid.best_params_['max_depth']}, ccp_alpha=0)")
    
    plot_tree(
        tree_pruned, filled=True, feature_names=None, 
        class_names=None, ax=axes[1], max_depth=3
    )
    axes[1].set_title(
        f"Árbol podado (max_depth={grid.best_params_['max_depth']}, "
        f"ccp_alpha={grid.best_params_['ccp_alpha']})"
    )
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar path de costo-complejidad
    path = tree_unpruned.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    # Evaluar diferentes niveles de poda
    trees = []
    for ccp_alpha in ccp_alphas:
        tree = DecisionTreeClassifier(random_state=random_state, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        trees.append(tree)
    
    # Comparar rendimiento
    train_scores = [tree.score(X_train, y_train) for tree in trees]
    test_scores = [tree.score(X_test, y_test) for tree in trees]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs alpha para diferentes niveles de poda")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return mejor_modelo, grid.cv_results_

# Ejemplo de uso (con datos hipotéticos):
# mejor_arbol, resultados = podar_arbol_decision(X, y)
```

#### **Técnica de Ensamblado:**

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def crear_ensamblado(X, y, random_state=42):
    """
    Crea y evalúa modelos ensamblados (voting y stacking)
    
    Parámetros:
    -----------
    X : array-like
        Características
    y : array-like
        Variable objetivo
    
    Retorna:
    --------
    dict
        Modelos ensamblados y sus métricas
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos base
    modelos_base = [
        ('lr', LogisticRegression(random_state=random_state)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ('svm', SVC(probability=True, random_state=random_state))
    ]
    
    # Crear ensamblado por votación
    voting_hard = VotingClassifier(estimators=modelos_base, voting='hard')
    voting_soft = VotingClassifier(estimators=modelos_base, voting='soft')
    
    # Crear ensamblado por stacking
    stacking = StackingClassifier(
        estimators=modelos_base,
        final_estimator=LogisticRegression(random_state=random_state)
    )
    
    # Entrenar modelos individuales y ensamblados
    resultados = {}
    modelos = {
        'Regresión Logística': modelos_base[0][1],
        'Random Forest': modelos_base[1][1],
        'SVM': modelos_base[2][1],
        'Voting (hard)': voting_hard,
        'Voting (soft)': voting_soft,
        'Stacking': stacking
    }
    
    for nombre, modelo in modelos.items():
        # Entrenar
        modelo.fit(X_train_scaled, y_train)
        
        # Predecir
        y_pred = modelo.predict(X_test_scaled)
        
        # Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        resultados[nombre] = {
            'modelo': modelo,
            'accuracy': accuracy,
            'reporte': report
        }
        
        print(f"\nResultados para {nombre}:")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Comparar resultados
    accuracies = [resultados[nombre]['accuracy'] for nombre in modelos.keys()]
    
    plt.figure(figsize=(12, 6))
    plt.bar(modelos.keys(), accuracies, color='steelblue', alpha=0.8)
    plt.xlabel('Modelo')
    plt.ylabel('Accuracy')
    plt.title('Comparación de Accuracy: Modelos individuales vs. Ensamblados')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return resultados

# Ejemplo de uso (con datos hipotéticos):
# resultados_ensamblados = crear_ensamblado(X, y)
```

#### **Técnicas específicas para redes neuronales:**

Para redes neuronales, las técnicas más efectivas incluyen:

1. **Dropout:** desactivar neuronas aleatoriamente durante el entrenamiento
2. **Regularización L1/L2:** añadir penalización a los pesos grandes
3. **Batch Normalization:** normalizar activaciones en capas intermedias
4. **Data Augmentation:** generar variaciones sintéticas de los datos
5. **Early Stopping:** detener entrenamiento cuando la validación empeora

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2

def crear_red_con_regularizacion(input_dim, hidden_layers=[64, 32], 
                                dropout_rate=0.3, l1_factor=0.0, 
                                l2_factor=0.01, use_batch_norm=True):
    """
    Crea una red neuronal con diferentes técnicas de regularización
    
    Parámetros:
    -----------
    input_dim : int
        Dimensión de entrada
    hidden_layers : list
        Lista con el número de neuronas por capa oculta
    dropout_rate : float
        Tasa de dropout (0-1)
    l1_factor : float
        Factor de regularización L1
    l2_factor : float
        Factor de regularización L2
    use_batch_norm : bool
        Si se usa normalización por lotes
    
    Retorna:
    --------
    modelo : Sequential
        Modelo de Keras con regularización
    """
    modelo = Sequential()
    
    # Capa de entrada
    modelo.add(Dense(
        hidden_layers[0], 
        input_dim=input_dim, 
        activation='relu',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    ))
    
    if use_batch_norm:
        modelo.add(BatchNormalization())
    
    if dropout_rate > 0:
        modelo.add(Dropout(dropout_rate))
    
    # Capas ocultas
    for units in hidden_layers[1:]:
        modelo.add(Dense(
            units, 
            activation='relu',
            kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
        ))
        
        if use_batch_norm:
            modelo.add(BatchNormalization())
        
        if dropout_rate > 0:
            modelo.add(Dropout(dropout_rate))
    
    # Capa de salida (para clasificación binaria)
    modelo.add(Dense(1, activation='sigmoid'))
    
    # Compilar
    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

# Configurar early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Ejemplo de uso:
# model = crear_red_con_regularizacion(input_dim=X_train.shape[1])
# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=32,
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=0
# )
```

#### **Consejos avanzados para combatir el sobreajuste:**

1. **Características fantasma:** Añadir variables aleatorias como control; si el modelo les da importancia, está sobreajustando.

2. **Validación temporal progresiva:** En datos temporales, usar ventanas de tiempo crecientes para validación.

3. **Pruebas de robustez:** Verificar estabilidad del modelo ante pequeñas perturbaciones en los datos.

4. **Descenso del gradiente estocástico con tasa de aprendizaje adaptativa:** Reducir la tasa de aprendizaje cuando el modelo converge.

5. **Comprensión del ruido inherente en los datos:** Establecer un "techo" teórico de rendimiento basado en la calidad de los datos.

> **Pregunta para reflexionar:** ¿Cómo sabes cuándo debes dejar de luchar contra el sobreajuste? ¿Qué señales indican que has alcanzado un balance adecuado entre sesgo y varianza?

---

### **4.3. Diagnosticar sesgo y varianza con curvas de aprendizaje**

**Problema:** Es difícil determinar si un modelo tiene problemas de sesgo (no aprende suficiente) o varianza (aprende demasiado), lo que lleva a estrategias de mejora ineficaces.

**Solución:** Utilizar curvas de aprendizaje para visualizar cómo el rendimiento del modelo cambia con el tamaño de los datos de entrenamiento, revelando patrones característicos de sesgo y varianza.

#### **Fundamentos de sesgo y varianza:**

El **sesgo** (bias) es el error por suposiciones simplificadas en el algoritmo, resultando en **subajuste**. La **varianza** es el error por sensibilidad a fluctuaciones en los datos, resultando en **sobreajuste**.

La **meta** es encontrar el balance entre ambos (trade-off):
- Modelos complejos ↓ sesgo ↑ varianza
- Modelos simples ↑ sesgo ↓ varianza

![Trade-off sesgo-varianza](https://i.imgur.com/placeholder_bias_variance.png)

#### **Interpretación de curvas de aprendizaje:**

| Patrón | Interpretación | Solución |
|--------|----------------|----------|
| **Alto error entrenamiento + alto error validación** | Alto sesgo (subajuste) | Aumentar complejidad del modelo |
| **Bajo error entrenamiento + alto error validación** | Alta varianza (sobreajuste) | Regularizar o reducir complejidad |
| **Errores altos + curvas separadas** | Sesgo y varianza altos | Más datos y/o mejor ingeniería de características |
| **Errores convergiendo a nivel alto** | Sesgo irreducible o problemas con datos | Revisar calidad de datos o cambiar de enfoque |

#### **Implementación de curvas de aprendizaje:**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def graficar_curva_aprendizaje(estimador, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    """
    Genera y visualiza curvas de aprendizaje para diagnosticar sesgo-varianza
    
    Parámetros:
    -----------
    estimador : estimator
        Modelo de scikit-learn
    X : array-like
        Características
    y : array-like
        Variable objetivo
    cv : int
        Número de folds para validación cruzada
    n_jobs : int
        Número de trabajos paralelos (-1 para todos los cores)
    train_sizes : array
        Proporciones del conjunto de entrenamiento a utilizar
    
    Retorna:
    --------
    dict
        Resultados detallados de las curvas de aprendizaje
    """
    # Calcular curvas de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(
        estimador, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='neg_mean_squared_error' if estimador._estimator_type == 'regressor' else 'accuracy'
    )
    
    # Calcular medias y desviaciones
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Para regresión, convertir MSE negativo a positivo
    if estimador._estimator_type == 'regressor':
        train_scores_mean = -train_scores_mean
        train_scores_std = train_scores_std
        test_scores_mean = -test_scores_mean
        test_scores_std = test_scores_std
        ylabel = 'Error Cuadrático Medio'
    else:
        ylabel = 'Accuracy'
    
    # Graficar curvas de aprendizaje
    plt.figure(figsize=(12, 6))
    plt.grid(True, alpha=0.3)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validación cruzada")
    
    plt.title(f"Curva de aprendizaje para {estimador.__class__.__name__}")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    
    # Añadir anotaciones de diagnóstico
    gap = np.abs(train_scores_mean[-1] - test_scores_mean[-1])
    train_level = train_scores_mean[-1]
    
    if estimador._estimator_type == 'regressor':
        # Para regresión: menor error es mejor
        if train_level > 0.1 and gap < 0.1:
            plt.annotate('Diagnóstico: Alto sesgo (subajuste)',
                        xy=(0.5, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        elif train_level < 0.1 and gap > 0.1:
            plt.annotate('Diagnóstico: Alta varianza (sobreajuste)',
                        xy=(0.5, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        elif train_level > 0.1 and gap > 0.1:
            plt.annotate('Diagnóstico: Alto sesgo y alta varianza',
                        xy=(0.5, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    else:
        # Para clasificación: mayor accuracy es mejor
        if train_level < 0.9 and gap < 0.1:
            plt.annotate('Diagnóstico: Alto sesgo (subajuste)',
                        xy=(0.5, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        elif train_level > 0.9 and gap > 0.1:
            plt.annotate('Diagnóstico: Alta varianza (sobreajuste)',
                        xy=(0.5, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        elif train_level < 0.9 and gap > 0.1:
            plt.annotate('Diagnóstico: Alto sesgo y alta varianza',
                        xy=(0.5, 0.1), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    # Información adicional
    print(f"Diagnóstico para {estimador.__class__.__name__}:")
    print(f"  Rendimiento final entrenamiento: {train_scores_mean[-1]:.4f}")
    print(f"  Rendimiento final validación: {test_scores_mean[-1]:.4f}")
    print(f"  Brecha (gap): {gap:.4f}")
    
    if estimador._estimator_type == 'regressor':
        if train_level > 0.1:
            print("  → Alto sesgo (subajuste): El modelo no está capturando patrones en los datos.")
            print("    Recomendación: Aumentar la complejidad del modelo o mejorar las características.")
        if gap > 0.1:
            print("  → Alta varianza (sobreajuste): El modelo no generaliza bien a datos nuevos.")
            print("    Recomendación: Aplicar regularización, reducir complejidad o aumentar datos.")
    else:
        if train_level < 0.9:
            print("  → Alto sesgo (subajuste): El modelo no está capturando patrones en los datos.")
            print("    Recomendación: Aumentar la complejidad del modelo o mejorar las características.")
        if gap > 0.1:
            print("  → Alta varianza (sobreajuste): El modelo no generaliza bien a datos nuevos.")
            print("    Recomendación: Aplicar regularización, reducir complejidad o aumentar datos.")
    
    # Retornar datos para análisis adicional
    return {
        'train_sizes': train_sizes,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'test_scores_mean': test_scores_mean,
        'test_scores_std': test_scores_std,
        'gap': gap,
        'train_level': train_level
    }

# Ejemplo de uso:
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('modelo', RandomForestClassifier(n_estimators=100, random_state=42))
# ])
# resultados = graficar_curva_aprendizaje(pipeline, X, y, cv=5)
```

#### **Visualizando el impacto de hiperparámetros en el sesgo-varianza:**

```python
def graficar_curva_validacion(estimador, X, y, param_name, param_range, cv=5, 
                             scoring='accuracy', log_scale=False):
    """
    Grafica curvas de validación para analizar el impacto de un hiperparámetro
    
    Parámetros:
    -----------
    estimador : estimator
        Modelo de scikit-learn
    X : array-like
        Características
    y : array-like
        Variable objetivo
    param_name : str
        Nombre del parámetro a variar
    param_range : array
        Valores del parámetro a evaluar
    cv : int
        Número de folds para validación cruzada
    scoring : str
        Métrica de evaluación
    log_scale : bool
        Si se usa escala logarítmica para el eje x
        
    Retorna:
    --------
    dict
        Resultados detallados de la curva de validación
    """
    train_scores, test_scores = validation_curve(
        estimador, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    
    # Calcular medias y desviaciones
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Para métricas negativas, convertir a positivas
    if scoring.startswith('neg_'):
        train_scores_mean = -train_scores_mean
        test_scores_mean = -test_scores_mean
        ylabel = scoring[4:].replace('_', ' ').title()
    else:
        ylabel = scoring.replace('_', ' ').title()
    
    # Graficar curva de validación
    plt.figure(figsize=(12, 6))
    plt.grid(True, alpha=0.3)
    
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Entrenamiento")
    plt.plot(param_range, test_scores_mean, 'o-', color="g", label="Validación cruzada")
    
    plt.title(f"Curva de validación: {param_name}")
    plt.xlabel(param_name)
    plt.ylabel(ylabel)
    
    if log_scale:
        plt.xscale('log')
    
    plt.legend(loc="best")
    
    # Encontrar el mejor valor del parámetro
    if scoring.startswith('neg_'):
        # Para métricas negativas, menor es mejor
        best_idx = np.argmin(test_scores_mean)
    else:
        # Para otras métricas, mayor es mejor
        best_idx = np.argmax(test_scores_mean)
    
    best_param = param_range[best_idx]
    best_score = test_scores_mean[best_idx]
    
    # Marcar el mejor punto
    plt.axvline(x=best_param, color='blue', linestyle='--', alpha=0.5)
    plt.scatter([best_param], [best_score], s=80, c='blue', marker='*', 
               label=f'Mejor: {best_param}')
    
    plt.annotate(f'Mejor {param_name}: {best_param}\nScore: {best_score:.4f}',
                xy=(best_param, best_score),
                xytext=(0, 20),
                textcoords='offset points',
                ha='center',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    # Diagnóstico
    print(f"Análisis de {param_name}:")
    print(f"  Mejor valor: {best_param} (score: {best_score:.4f})")
    
    # Analizar tendencia para diagnosticar sesgo-varianza
    trend_direction = "aumenta" if test_scores_mean[-1] > test_scores_mean[0] else "disminuye"
    gap_first = abs(train_scores_mean[0] - test_scores_mean[0])
    gap_last = abs(train_scores_mean[-1] - test_scores_mean[-1])
    
    print(f"  Rendimiento en validación {trend_direction} a medida que {param_name} crece")
    print(f"  Brecha inicio: {gap_first:.4f}, Brecha final: {gap_last:.4f}")
    
    if scoring.startswith('neg_'):
        # Para métricas de error, menor es mejor
        if trend_direction == "disminuye" and gap_last < gap_first:
            print("  → Reducción de sesgo: El modelo se ajusta mejor con valores más altos.")
        elif trend_direction == "aumenta" and gap_last > gap_first:
            print("  → Aumento de varianza: Posible sobreajuste con valores más altos.")
    else:
        # Para métricas de rendimiento, mayor es mejor
        if trend_direction == "aumenta" and gap_last < gap_first:
            print("  → Reducción de sesgo: El modelo se ajusta mejor con valores más altos.")
        elif trend_direction == "disminuye" and gap_last > gap_first:
            print("  → Aumento de varianza: Posible sobreajuste con valores más altos.")
    
    return {
        'param_range': param_range,
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'test_scores_mean': test_scores_mean,
        'test_scores_std': test_scores_std,
        'best_param': best_param,
        'best_score': best_score
    }

# Ejemplo de uso:
# from sklearn.ensemble import RandomForestClassifier
# modelo = RandomForestClassifier(random_state=42)
# param_range = [5, 10, 20, 30, 50, 100, 200]
# resultados = graficar_curva_validacion(
#     modelo, X, y, 'n_estimators', param_range, 
#     scoring='accuracy', log_scale=False
# )
```

#### **Estrategias correctivas basadas en el diagnóstico:**

| Problema diagnosticado | Soluciones recomendadas |
|------------------------|-------------------------|
| **Alto sesgo (subajuste)** | • Aumentar complejidad del modelo<br>• Añadir características/polinomios<br>• Reducir regularización<br>• Probar algoritmos más potentes |
| **Alta varianza (sobreajuste)** | • Aumentar regularización<br>• Simplificar el modelo<br>• Recolectar más datos<br>• Técnicas de ensamblado<br>• Reducir dimensionalidad |
| **Alto sesgo y varianza** | • Mejor ingeniería de características<br>• Aumentar calidad de datos<br>• Validación cruzada para selección de modelos |

> **Ejercicio práctico:** Selecciona un conjunto de datos y un algoritmo. Genera curvas de aprendizaje variando sistemáticamente la cantidad de datos de entrenamiento. ¿Puedes identificar si el modelo sufre de alto sesgo, alta varianza o ambos? Implementa una estrategia correctiva y verifica si mejora el rendimiento.

---

### **4.4. Modelar datasets a gran escala**

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

## **5. Mejores prácticas en la etapa de despliegue y monitoreo**

Después de preparar los datos, generar el conjunto de entrenamiento y entrenar el modelo, llega el momento de **desplegar el sistema**. Aquí nos aseguramos de que los modelos funcionen bien en producción, se actualicen si es necesario y sigan ofreciendo valor real.

---

### **5.1. Guardar, cargar y reutilizar modelos**

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

### **5.2. Monitorear el rendimiento del modelo**

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

### **5.3. Actualizar los modelos regularmente**

Con el tiempo, los datos pueden cambiar (fenómeno conocido como *data drift*). Si el rendimiento se deteriora:

* **Monitorea constantemente**: si las métricas bajan, es momento de actuar.
* **Actualizaciones programadas**: según frecuencia de cambios en los datos.
* **Aprendizaje en línea (online learning)**: para modelos como regresión con SGD o Naïve Bayes, que se pueden actualizar sin reentrenar.
* **Control de versiones**: tanto de modelos como de datasets.
* **Auditorías regulares**: revisa si las métricas, objetivos de negocio o datos han cambiado.

> 📌 Monitorear es un proceso continuo, no algo que se hace una sola vez.

---

## **6. Resumen**

Esta guía te prepara para resolver problemas reales de machine learning. Repasamos el flujo de trabajo típico:

1. Preparación de datos
2. Generación del conjunto de entrenamiento
3. Entrenamiento, evaluación y selección de modelos
4. Despliegue y monitoreo

Para cada etapa, detallamos tareas, desafíos comunes y **21 mejores prácticas**.

> ✅ **La mejor práctica de todas es practicar.**
> Empieza un proyecto real y aplica lo que has aprendido.