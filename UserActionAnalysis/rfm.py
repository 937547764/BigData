from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, max, count, datediff, lit, expr
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StringType
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName("RFAnalysis").getOrCreate()

file_path = 'hdfs://namenode:9000//user_log_format1.csv'
data = spark.read.csv(file_path, header=True, inferSchema=True)

purchase_data = data.filter(col("action_type") == 2)
purchase_data = purchase_data.filter(col("time_stamp") < 1111)

purchase_data = purchase_data.withColumn("formatted_time_stamp", expr("lpad(time_stamp, 4, '0')"))
purchase_data = purchase_data.withColumn("date", to_date(expr("concat('2016', formatted_time_stamp)"), "yyyyMMdd"))

rf_data = purchase_data.groupBy("user_id").agg(max("date").alias("LastPurchaseDate"), count("user_id").alias("Frequency"))

reference_date = lit("2016-11-11")
rf_data = rf_data.withColumn("Recency", datediff(reference_date, col("LastPurchaseDate")))

recency_stats = rf_data.describe("Recency")
recency_stats.show()

frequency_stats = rf_data.describe("Frequency")
frequency_stats.show()

rf_data_pd = rf_data.toPandas()

sns.set(style="whitegrid")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.distplot(rf_data_pd['Recency'], bins=40, kde=False)
plt.title('Recency Distribution')

plt.subplot(1, 2, 2)
sns.distplot(rf_data_pd[rf_data_pd['Frequency'] < 40]['Frequency'], bins=40, kde=False)
plt.title('Frequency Distribution')

plt.savefig('Recency_and_Frequency.png')

quantiles = rf_data_pd.quantile(q=[0.25, 0.5, 0.75]).to_dict()

def RScore(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

def FScore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

RScore_udf = udf(lambda x: RScore(x, 'Recency', quantiles), IntegerType())
FScore_udf = udf(lambda x: FScore(x, 'Frequency', quantiles), IntegerType())

rf_data = rf_data.withColumn('R', RScore_udf('Recency'))
rf_data = rf_data.withColumn('F', FScore_udf('Frequency'))

def assign_user_type(R, F):
    if R >= 3 and F >= 3:
        return 'Valuable User'
    elif R >= 3 and F < 3:
        return 'Developing User'
    elif R < 3 and F >= 3:
        return 'Maintain User'
    else:
        return 'Retain User'

assign_user_type_udf = udf(assign_user_type, StringType())

rf_data = rf_data.withColumn('User_Type', assign_user_type_udf('R', 'F'))

user_type_counts = rf_data.groupBy('User_Type').count()

user_type_counts_pd = user_type_counts.toPandas()
print(user_type_counts_pd)

plt.figure(figsize=(10, 6))
barplot = user_type_counts_pd.plot(kind='bar', color='skyblue')

for bar in barplot.patches:
    barplot.annotate(format(bar.get_height()),
                     (bar.get_x() + bar.get_width() / 2,
                      bar.get_height()), ha='center', va='center',
                     size=10, xytext=(0, 8),
                     textcoords='offset points')

plt.title('User Type Count Statistics')
plt.xlabel('User Type')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('user_types_chart.png')