# Databricks notebook source
import pyspark

# COMMAND ----------

# EXERCISE 1

print(sc.getConf().get('spark.executor.memory'))
# About 8GB of memory. There are 15GB of memory in the default cluster and 2 cores--so this is half the memory per core.

print(sc.getConf().get('spark.executor.tempDirectory'))
# This is NOT part of the Databricks file system! (At least not formally.) This is a "local" disk for temporary storage.

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

row1 = ['Alice', 1, 8.9]
row2 = ['Bob', 2, 3.3]
row3 = ['Charlie', 3, 5.1]

df = spark.createDataFrame(
    [row1, row2, row3],
    ['name', 'id', 'rating']
)

# COMMAND ----------

df.show()

# COMMAND ----------

df.summary().show()

# COMMAND ----------

from pyspark.sql.functions import mean
df.select(mean('rating')).show()

# COMMAND ----------

# EXERCISE 2
# 2(a)
from pyspark.sql.functions import sqrt
df.select(sqrt('rating')).show()

# 2(b)
# Expect it to fail, because 'name' is a string column and strings don't have a square root.
df.select(sqrt('name')).show()
# It doesn't throw an error! Just fails silently and returns a 'null' value.

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df2 = df.withColumn('rating', df['rating'].cast('int'))

# COMMAND ----------

# EXERCISE 3
# 3(a)
df_3a = df.withColumn('id', df['id'].cast('double'))
df_3a.show()
# Does not change any values, does change the display to look like a float.

# 3(b)
df_3b = df.withColumn('rating', df['rating'].cast('int'))
df_3b.show()
# Rounds all values down

# 3(c)
df_3c = df.withColumn('rating', df['rating'].cast('string'))
df_3c.select(mean('rating')).show()
# Expect this to fail! (or produce null). But it reinterprets the field as numeric because all values can be cast as numbers, and calculates the mean anyway.

# COMMAND ----------

# EXERCISE 4
patient_info = spark.read.format('csv') \
   .option('header', 'true') \
   .load('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/PatientInfo.csv')
weather = spark.read.format('csv') \
    .option('header', 'true') \
    .load('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/Weather.csv')
search_trend = spark.read.format('csv') \
    .option('header', 'true') \
    .load('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/SearchTrend.csv')

patient_info.show()
patient_info.printSchema()

weather.show()
weather.printSchema()

search_trend.show()
search_trend.printSchema()

# Reads pretty much everything as strings...due to presence of null values.

# COMMAND ----------

# EXERCISE 5
odyssey = spark.read.text('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/Odyssey.txt')
odyssey.show()
# Rows are lines in the file
# Only one column: the text of the line

# COMMAND ----------

# EXERCISE 6
# 6(a)
new_patient_info = patient_info \
    .withColumn('symptom_onset_date', patient_info['symptom_onset_date'].cast('date')) \
    .withColumn('confirmed_date', patient_info['confirmed_date'].cast('date')) \
    .withColumn('released_date', patient_info['released_date'].cast('date')) \
    .withColumn('deceased_date', patient_info['deceased_date'].cast('date'))

# 6(b)
root_path = 'dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/'
new_patient_info.repartition(1).write.options(header='true').csv(root_path+'new_patient_info_csv.csv')
new_patient_info.write.parquet(root_path+'new_patient_info_parquet')

# 6(c)
new_patient_info_csv_read = spark.read.format('csv') \
   .option('header', 'true') \
   .load(root_path+'new_patient_info_csv.csv')
new_patient_info_csv_read.printSchema()

new_patient_info_parquet_read = spark.read.parquet(root_path+'new_patient_info_parquet')
new_patient_info_parquet_read.printSchema()

# Writing to CSV loses the date recast
# Writing to Parquet keeps the types as recasted

# COMMAND ----------

# EXERCISE 7
# 7(a)
sc.stop()
odyssey = spark.read.text('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/Odyssey.txt')
odyssey.show()
# Doesn't work! Server hangs.

# COMMAND ----------

# 7(b)
print(3+4)
# This also doesn't work! It seems like it should. But this is because the entire environment on Databricks is running with Spark; when you stop the SparkContext, you stop EVERYTHING. If you were running on your own machine this would execute just fine.

# COMMAND ----------


