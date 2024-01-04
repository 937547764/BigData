from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("UserClustering").getOrCreate()

product_data = spark.read.csv("hdfs://namenode:9000/demo.csv", header=True, inferSchema=True).na.drop()
user_data = spark.read.csv("hdfs://namenode:9000/user_info_format1.csv", header=True, inferSchema=True).na.drop()

data = product_data.join(user_data, "user_id", "left").na.fill(0)

features = data.select('user_id', 'age_range', 'gender', 'item_id', 'cat_id', 'seller_id', 'brand_id')

feature_columns = features.columns[1:]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
features_assembled = assembler.transform(features)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(features_assembled)
features_scaled = scaler_model.transform(features_assembled)

kmeans = KMeans(k=5, seed=42, featuresCol="scaled_features", predictionCol="Cluster")
model = kmeans.fit(features_scaled)
clusters = model.transform(features_scaled)

evaluator = ClusteringEvaluator(featuresCol='scaled_features', predictionCol='Cluster')
silhouette_avg = evaluator.evaluate(clusters)
print("Silhouette Score: {}".format(silhouette_avg))

# data_with_clusters = data.join(clusters.select("user_id", "Cluster"), "user_id", "left")
# pandas_df = data_with_clusters.toPandas()
# tsne = TSNE(n_components=2, random_state=42)
# tsne_results = tsne.fit_transform(pandas_df['scaled_features'].tolist())
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=pandas_df['Cluster'], palette='viridis', alpha=0.7, s=5)
# plt.title('t-SNE Visualization of Clusters')
# plt.show()
# plt.savefig("./User.png")

spark.stop()
