## 搭建虚拟化环境
### 拉取镜像并创建容器
cd environment
docker-compose up -d
### 配置容器环境
#### Spark Master
docker exec -it master bash
rm /etc/apt/sources.list
exit
#### 宿主机
docker cp sources.list master:/etc/apt/sources.list
#### Spark Master
docker exec -it master bash
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 871920D1991BC93C
apt-get update
apt-get install python-tk
apt-get install vim
vim ~/.bashrc (添加 alias python='/usr/bin/python2.7')
source ~/.bashrc
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py | python2.7
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
## 数据分析
### 复购预测
#### 上传数据至hdfs
##### 宿主机
cd prediction
docker cp data.csv namenode:/
##### namenode
docker exec -it namenode bash
hdfs dfs -put /data.csv /
exit
#### 拷贝代码至Spark Master并执行
##### 宿主机
docker cp RepurchasePrediction.py master:/
##### master
docker exec -it master bash
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /RepurchasePrediction.py
exit
### 聚类分析及关联规则挖掘
#### 上传数据至hdfs
##### 宿主机
cd clustering
docker cp demo.csv namenode:/
docker cp user_info_format1.csv namenode:/
##### namenode
docker exec -it namenode bash
hdfs dfs -put /demo.csv /
hdfs dfs -put /user_info_format1.csv /
exit
#### 拷贝代码至Spark Master并执行
##### 宿主机
docker cp UserClustering.py master:/
docker cp ProductClustering.py master:/
docker cp ProductAssociation.py master:/
##### master
docker exec -it master bash
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /UserClustering.py
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /ProductClustering.py
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /ProductAssociation.py
exit