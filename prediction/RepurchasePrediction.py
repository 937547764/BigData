from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("ML").getOrCreate()

data = spark.read.csv("hdfs://namenode:9000/data.csv", header=True, inferSchema=True)

feature_columns = [co for co in data.columns if co != "label"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(data)
data = scaler_model.transform(data)

train_data, test_data = data.randomSplit([0.75, 0.25], seed=42)

lr = LogisticRegression(featuresCol="scaled_features", labelCol="label", maxIter=20)
dt = DecisionTreeClassifier(featuresCol="scaled_features", labelCol="label", maxDepth=20)
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label", maxDepth=20, numTrees=11)

lr_pipeline = Pipeline(stages=[lr])
dt_pipeline = Pipeline(stages=[dt])
rf_pipeline = Pipeline(stages=[rf])

lr_model = lr_pipeline.fit(train_data)
dt_model = dt_pipeline.fit(train_data)
rf_model = rf_pipeline.fit(train_data)

lr_predictions = lr_model.transform(test_data)
dt_predictions = dt_model.transform(test_data)
rf_predictions = rf_model.transform(test_data)

evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
lr_auc = evaluator.evaluate(lr_predictions)
dt_auc = evaluator.evaluate(dt_predictions)
rf_auc = evaluator.evaluate(rf_predictions)


def plot_roc_curve(labels, scores, model_name):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='%s (AUC = %.2f)' % (model_name, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s ROC Curve' % model_name)
    plt.legend(loc='lower right')
    plt.savefig("%s_ROC_Curve.png" % model_name)
    plt.show()


lr_scores = np.array(lr_predictions.select("probability").collect())[:, 0, 1]
plot_roc_curve(np.array(test_data.select("label").collect()), lr_scores, "LogisticRegression")

dt_scores = np.array(dt_predictions.select("probability").collect())[:, 0, 1]
plot_roc_curve(np.array(test_data.select("label").collect()), dt_scores, "DecisionTree")

rf_scores = np.array(rf_predictions.select("probability").collect())[:, 0, 1]
plot_roc_curve(np.array(test_data.select("label").collect()), rf_scores, "RandomForest")


def calculate_metrics(predictions):
    tp = predictions.filter("label = 1 AND prediction = 1").count()
    fp = predictions.filter("label = 0 AND prediction = 1").count()
    tn = predictions.filter("label = 0 AND prediction = 0").count()
    fn = predictions.filter("label = 1 AND prediction = 0").count()

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    ret = {}
    ret['accuracy'] = round(accuracy, 2)
    ret['precision'] = round(precision, 2)
    ret['recall'] = round(recall, 2)
    ret['f1-score'] = round(f1, 2)

    return ret


lr_metrics = calculate_metrics(lr_predictions)
dt_metrics = calculate_metrics(dt_predictions)
rf_metrics = calculate_metrics(rf_predictions)

output_file_path = './result'
with open(output_file_path, 'w') as f:
    f.write("\n")
    f.write("Logistic Regression Metrics:\n" + str(lr_metrics) + "\n")
    f.write("Decision Tree Metrics:\n" + str(dt_metrics) + "\n")
    f.write("Random Forest Metrics:\n" + str(rf_metrics) + "\n")
