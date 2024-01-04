## 搭建虚拟化环境
### 拉取镜像并创建容器
cd environment<br>
docker-compose up -d
### 配置容器环境
#### Spark Master
docker exec -it master bash<br>
rm /etc/apt/sources.list<br>
exit
#### 宿主机
docker cp sources.list master:/etc/apt/sources.list
#### Spark Master
docker exec -it master bash<br>
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 871920D1991BC93C<br>
apt-get update<br>
apt-get install python-tk<br>
apt-get install vim<br>
vim ~/.bashrc (添加 alias python='/usr/bin/python2.7')<br>
source ~/.bashrc<br>
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py | python2.7<br>
pip install numpy<br>
pip install matplotlib<br>
pip install pandas<br>
pip install scikit-learn<br>
pip install seaborn
## 数据分析
### 复购预测
#### 上传数据至HDFS
##### 宿主机
cd prediction<br>
docker cp data.csv namenode:/
##### Namenode
docker exec -it namenode bash<br>
hdfs dfs -put /data.csv /<br>
exit
#### 拷贝代码至Spark Master并执行
##### 宿主机
docker cp RepurchasePrediction.py master:/
##### Spark Master
docker exec -it master bash<br>
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /RepurchasePrediction.py<br>
exit
### 聚类分析及关联规则挖掘
#### 上传数据至HDFS
##### 宿主机
cd clustering<br>
docker cp demo.csv namenode:/<br>
docker cp user_info_format1.csv namenode:/
##### Namenode
docker exec -it namenode bash<br>
hdfs dfs -put /demo.csv /<br>
hdfs dfs -put /user_info_format1.csv /<br>
exit
#### 拷贝代码至Spark Master并执行
##### 宿主机
docker cp UserClustering.py master:/<br>
docker cp ProductClustering.py master:/<br>
docker cp ProductAssociation.py master:/
##### Spark Master
docker exec -it master bash<br>
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /UserClustering.py<br>
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /ProductClustering.py<br>
spark-submit --conf spark.pyspark.python=/usr/bin/python2.7 --master spark://master:7077 /ProductAssociation.py<br>
exit