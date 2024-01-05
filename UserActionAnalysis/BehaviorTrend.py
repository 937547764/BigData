from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, expr
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName("BehaviorAnalysisOverTime").getOrCreate()

file_path = 'hdfs://namenode:9000//user_log_format1.csv'
user_data = spark.read.csv(file_path, header=True, inferSchema=True)

user_data = user_data.filter(col("time_stamp") > 1010)
user_data = user_data.withColumn("formatted_time_stamp", expr("lpad(time_stamp, 4, '0')"))
user_data = user_data.withColumn("date", to_date(expr("concat('2016', formatted_time_stamp)"), "yyyyMMdd"))


action_counts_date = user_data.groupBy("date", "action_type").count()
action_counts_date_pandas = action_counts_date.toPandas()
action_counts_date_pandas_pivot = action_counts_date_pandas.pivot(index='date', columns='action_type', values='count').fillna(0)




sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))
action_counts_date_pandas_pivot.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Daily User Behavior Trends')
plt.xlabel('Date (MM-DD)')
plt.ylabel('Number of Actions')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)
plt.legend(['Click', 'Add to Cart', 'Purchase', 'Add to Favorites'], title='Action Type')
plt.grid(True)
plt.tight_layout()
plt.savefig('BehaviorTrendAnalysisOverTime.png')
