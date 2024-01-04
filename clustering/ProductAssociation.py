from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_set, col

spark = SparkSession.builder.appName("ProductAssociation").getOrCreate()

file_path = "hdfs://namenode:9000/demo.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

data = df.select("user_id", "item_id")
baskets = data.groupBy("user_id").agg(collect_set("item_id").alias("items"))

fp_growth = FPGrowth(itemsCol="items", minSupport=0.01, minConfidence=0.1)
model = fp_growth.fit(baskets)

frequent_itemsets = model.freqItemsets
rules = model.associationRules
print("\nFrequent Itemsets:")
frequent_itemsets.show(truncate=False)
print("\nAssociation Rules:")
rules.show(truncate=False)

spark.stop()
