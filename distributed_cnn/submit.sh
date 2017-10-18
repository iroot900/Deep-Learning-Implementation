#export SPARK_HOME=
#export HADOOP_HOME=

export PATH=$SPARK_HOME/bin:$PATH

export PYSPARK_PYTHON=./anaconda_remote/bin/python

cmd2="${SPARK_HOME}/bin/spark-submit \
  --master \
  yarn-cluster \
  --deploy-mode \
  cluster \
  --name 'dist_conv' \
  --class clear \
  --executor-memory=4g \
  --num-executors=40 \
  --executor-cores=2 \
  --driver-memory=4g \
  --queue \
  default \
  --conf \
  spark.admin.acls=* \
  --conf \
  spark.yarn.security.tokens.hbase.enabled=false \
  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./anaconda_remote/bin/python \
  --conf spark.dynamicAllocation.enabled=true \
  --archives hdfs://somepath/anaconda2.zip#anaconda_remote \
  --py-files layers.py,convnet.py \
  --verbose \
  $1 \
  "

$cmd2 
