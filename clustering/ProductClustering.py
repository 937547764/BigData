from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.sql import functions as F
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("ProductClustering").getOrCreate()

file_path = "hdfs://namenode:9000/demo.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)
data = data.na.drop()

features = data.select('user_id', 'item_id', 'cat_id', 'seller_id', 'brand_id')
if 'user_id' in data.columns and 'item_id' in data.columns:
    purchase_freq = data.groupBy('user_id', 'item_id').agg(F.count('*').alias('freq'))
    features = features.join(purchase_freq, ['user_id', 'item_id'], 'left_outer').na.fill(0)

feature_columns = ['cat_id', 'seller_id', 'brand_id', 'freq']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
features = assembler.transform(features)
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=False)
scaler_model = scaler.fit(features)
features_scaled = scaler_model.transform(features).select('user_id','scaled_features')

kmeans = KMeans(k=8, seed=42, featuresCol='scaled_features', predictionCol='Cluster')
model = kmeans.fit(features_scaled)
clusters = model.transform(features_scaled)

evaluator = ClusteringEvaluator(featuresCol='scaled_features', predictionCol='Cluster')
silhouette_avg = evaluator.evaluate(clusters)
print("Silhouette Score: {}".format(silhouette_avg))

# data_with_clusters = data.join(clusters.select('user_id','Cluster'), 'user_id', 'left_outer')#.drop(clusters['user_id'])
# features_pd = data_with_clusters.select('scaled_features').toPandas()
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(features_pd['scaled_features'].tolist())
# data_pd = data_with_clusters.toPandas()
# data_pd['tsne_x'] = tsne_results[:, 0]
# data_pd['tsne_y'] = tsne_results[:, 1]
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='tsne_x', y='tsne_y', hue='Cluster', palette='viridis', data=data_pd, alpha=0.7, s=5)
# plt.title('t-SNE Visualization of Clusters')
# plt.show()
# plt.savefig("./Product.png")

spark.stop()
