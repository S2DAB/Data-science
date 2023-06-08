import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

# Cargar los datos desde el archivo XLSX
data = pd.read_excel('base.xlsx')

# Visualizar los primeros registros de los datos
print(data.head())

# Obtener estadísticas básicas de la campaña
campaign_stats = data.describe()
print(campaign_stats)

# Calcular el total gastado en la campaña
total_spent = data['Importe gastado (MXN)'].sum()
print("Total gastado en la campaña: MXN", total_spent)

# Calcular el alcance promedio de la campaña
average_reach = data['Alcance'].mean()
print("Alcance promedio de la campaña:", average_reach)

# Calcular el costo por resultado promedio
average_cost_per_result = data['Costo por resultados'].mean()
print("Costo por resultado promedio: MXN", average_cost_per_result)

# Graficar el gasto por resultado
plt.figure(figsize=(8, 6))
plt.bar(data['Nombre de la campaña'], data['Importe gastado (MXN)'])
plt.xticks(rotation=90)
plt.xlabel('Campaña')
plt.ylabel('Gasto (MXN)')
plt.title('Gasto por Resultado')
plt.show()

# Graficar el alcance de la campaña
plt.figure(figsize=(8, 6))
plt.plot(data['Nombre de la campaña'], data['Alcance'], marker='o')
plt.xticks(rotation=90)
plt.xlabel('Campaña')
plt.ylabel('Alcance')
plt.title('Alcance de la Campaña')
plt.show()

# Seleccionar las características relevantes para la clusterización
features = ['Impresiones', 'Importe gastado (MXN)']

# Preprocesamiento de datos: estandarización
data_scaled = (data[features] - data[features].mean()) / data[features].std()

# Definir el número de clusters
n_clusters = 3

# Aplicar el algoritmo de K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(data_scaled)

# Obtener las etiquetas de cluster asignadas a cada punto
cluster_labels = kmeans.labels_

# Agregar las etiquetas al DataFrame original
data['Cluster'] = cluster_labels

# Graficar los resultados de la clusterización basada en centroides
plt.figure(figsize=(8, 6))
plt.scatter(data['Impresiones'], data['Importe gastado (MXN)'], c=data['Cluster'])
plt.xlabel('Impresiones')
plt.ylabel('Importe gastado (MXN)')
plt.title('Clusterización basada en Centroides')
plt.show()

# Realizar el agrupamiento utilizando DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data_scaled)

# Obtener las etiquetas de cluster asignadas a cada punto
cluster_labels_dbscan = dbscan.labels_

# Agregar las etiquetas al DataFrame original
data['Cluster_DBSCAN'] = cluster_labels_dbscan

# Graficar los resultados del agrupamiento DBSCAN
# Graficar los resultados del agrupamiento DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(data['Impresiones'], data['Importe gastado (MXN)'], c=data['Cluster_DBSCAN'])
plt.xlabel('Impresiones')
plt.ylabel('Importe gastado (MXN)')
plt.title('Agrupamiento DBSCAN')
plt.show()

# Seleccionar las características relevantes para la agrupación jerárquica
features = ['Impresiones', 'Importe gastado (MXN)']

# Obtener los datos a utilizar en la agrupación jerárquica
data_for_clustering = data[features]

# Calcular la matriz de enlace utilizando el método de enlace completo
linkage_matrix = linkage(data_for_clustering, method='complete')

# Graficar el dendrograma resultante
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix)
plt.title('Dendrograma de la Agrupación Jerárquica')
plt.xlabel('Índices de los Datos')
plt.ylabel('Distancia')
plt.show()

# Seleccionar la característica de interés para la gráfica de distribución
feature = 'Importe gastado (MXN)'

# Graficar un histograma
plt.figure(figsize=(8, 6))
sns.histplot(data[feature], kde=False)
plt.xlabel(feature)
plt.ylabel('Frecuencia')
plt.title('Distribución de la Característica (Histograma)')
plt.show()

# Graficar un gráfico de densidad
plt.figure(figsize=(8, 6))
sns.kdeplot(data[feature])
plt.xlabel(feature)
plt.ylabel('Densidad')
plt.title('Distribución de la Característica (Densidad)')
plt.show()
# Contar el número de puntos en cada cluster de K-means
cluster_counts = data['Cluster'].value_counts()

# Graficar el porcentaje de puntos en cada cluster de K-means en un gráfico de pastel
plt.figure(figsize=(8, 6))
plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%')
plt.title('Porcentaje de Puntos en cada Cluster (K-means)')
plt.show()

# Contar el número de puntos en cada cluster de DBSCAN
dbscan_cluster_counts = data['Cluster_DBSCAN'].value_counts()

# Graficar el porcentaje de puntos en cada cluster de DBSCAN en un gráfico de pastel
plt.figure(figsize=(8, 6))
plt.pie(dbscan_cluster_counts, labels=dbscan_cluster_counts.index, autopct='%1.1f%%')
plt.title('Porcentaje de Puntos en cada Cluster (DBSCAN)')
plt.show()
# Ordenar los datos por alcance de campaña de forma descendente
sorted_data = data.sort_values(by='Alcance', ascending=False)

# Crear una lista con los nombres de las campañas en orden jerárquico
campaign_names = sorted_data['Nombre de la campaña'].tolist()

# Ordenar los datos por alcance de campaña de forma descendente
sorted_data = data.sort_values(by='Alcance', ascending=False)

# Crear una tabla con el orden jerárquico del alcance de las campañas
table_data = sorted_data[['Nombre de la campaña', 'Alcance']].reset_index(drop=True)
table_data.index += 1  # Ajustar los índices comenzando desde 1

# Mostrar la tabla
print(table_data)
