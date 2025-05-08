
# 📊 Segmentación de Clientes con Machine Learning

Este proyecto implementa técnicas de **clustering no supervisado** para segmentar clientes utilizando el dataset *Mall Customers*. Se aplican los algoritmos de **K-Means** y **Clustering Jerárquico (Agglomerative)**, incluyendo un análisis completo y visualización de los resultados para ayudar a la toma de decisiones estratégicas en marketing.

---

## 📁 Estructura del Proyecto

- `Hierarchical.ipynb`: Notebook completo que realiza segmentación con Clustering Jerárquico.
- `K_Means.ipynb`: Notebook completo que realiza segmentación con K-Means.
- `Mall_Customers.csv`: Dataset original utilizado en los análisis.

---

## 🧠 Objetivos del Análisis

1. Identificar patrones de comportamiento entre los clientes.
2. Segmentar la base de clientes para mejorar estrategias de negocio.
3. Comparar el desempeño de distintos algoritmos de clustering.
4. Visualizar gráficamente los grupos detectados.

---

## 🔍 Contenido de los Notebooks

Cada notebook incluye:

1. **Análisis Exploratorio de Datos (EDA):**
   - Estadísticas descriptivas.
   - Visualización de distribuciones y correlaciones.

2. **Preprocesamiento:**
   - Limpieza de datos.
   - Codificación de variables categóricas.
   - Escalamiento de características.

3. **Selección de Características Relevantes.**

4. **Entrenamiento del Modelo:**
   - Para K-Means: elección del número óptimo de clusters con el método del codo.
   - Para Clustering Jerárquico: dendrograma para decidir el número de clusters.

5. **Evaluación del Modelo:**
   - Coeficiente de Silhouette.
   - Índice de Calinski-Harabasz.

6. **Visualización de Clusters:**
   - Gráficos 2D con color por grupo.
   - (Opcional) Reducción de dimensiones con PCA.

7. **Interpretación de Resultados:**
   - Análisis de cada clúster.
   - Implicaciones de negocio.

---

## 📦 Requisitos

- Python 3.x  
- Librerías:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `scipy`

Instala dependencias con:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## 📌 Dataset

**Mall_Customers.csv**  
- 200 registros de clientes con las siguientes variables:
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

Fuente: [Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

---

## 🧾 Conclusión

Este proyecto demuestra cómo el análisis de clústeres puede revelar segmentos útiles de clientes que no son evidentes con métodos tradicionales. Estas técnicas son valiosas para:
- Personalización de campañas de marketing.
- Optimización de recursos.
- Identificación de oportunidades comerciales.
