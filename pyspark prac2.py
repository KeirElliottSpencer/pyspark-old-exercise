# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

filepath = "dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/"
patient_info = spark.read.format("csv").option("header", "true").load(filepath+"PatientInfo.csv")
weather = spark.read.format("csv").option("header", "true").load(filepath+"Weather.csv")
search_trend = spark.read.format("csv").option("header", "true").load(filepath+"SearchTrend.csv")

# COMMAND ----------

# EXERCISE 1
# (a) 
def count_it(tbl, lbl):
    print('{0:15s}: {1:,} rows'.format(lbl, tbl.count()))
count_it(patient_info, 'patient_info')
count_it(weather, 'weather')
count_it(search_trend, 'search_trend')
# (b)
patient_info.show()
weather.show()
search_trend.show()
# (c)
patient_info.printSchema()
weather.printSchema()
search_trend.printSchema()
# A lot of these numeric fields were read in as strings and likely need to be converted.
# Also a lot of date/time stamps.

# COMMAND ----------

# EXERCISE 2
# first need to re-cast all the search trend value columns to doubles
search_trend = (search_trend
                .withColumn('cold', search_trend.cold.cast('double'))
                .withColumn('flu', search_trend.flu.cast('double'))
                .withColumn('pneumonia', search_trend.pneumonia.cast('double'))
                .withColumn('coronavirus', search_trend.coronavirus.cast('double'))
               )
# Now we can do the actual calculation
def corr_it(col1, col2): 
    print('{0:12s} <-> {1:12s} :: {2:.6f}'.format(col1, col2, search_trend.corr(col1, col2)))
corr_it('cold', 'flu')
corr_it('cold', 'pneumonia')
corr_it('cold', 'coronavirus')
corr_it('flu', 'pneumonia')
corr_it('flu', 'coronavirus')
corr_it('pneumonia', 'coronavirus')

# COMMAND ----------

# EXERCISE 3
df1 = search_trend.where(search_trend.date < '2019-03-08')
# (a)
print('DF1 is:', df1)
# This is just a descriptor! Not the actual data. That hasn't been evaluated yet.
# (b)
df1.show()
df1.collect()

# COMMAND ----------

# EXERCISE 4
pre_2020 = search_trend.where(search_trend.date < '2020-01-01')
post_2020 = search_trend.where(search_trend.date >= '2020-01-01')
# (a)
def part_a(col):
    pre_quantiles = pre_2020.approxQuantile(col, [0.25, 0.5, 0.9], 0.01)
    post_quantiles = post_2020.approxQuantile(col, [0.25, 0.5, 0.9], 0.01)
    print('{0:12s} | {1:.5f}/{2:.5f}/{3:.5f} | {4:.5f}/{5:.5f}/{6:.5f}'.format(
        col,
        pre_quantiles[0], pre_quantiles[1], pre_quantiles[2],
        post_quantiles[0], post_quantiles[1], post_quantiles[2]
    ))
print('--- APPROX QUANTILES ---')
part_a('cold')
part_a('flu')
part_a('pneumonia')
part_a('coronavirus')
# Coronavirus jumps A LOT, unsurprisingly. Some change on the others but only at the high end.
# BONUS: exact calculation would require combining across all partitions, which is slooooowwwwww

# (b)
print('\n---PAIRWISE CORRELATIONS ---')
def part_b(col1, col2):
    pre = pre_2020.corr(col1, col2)
    post = post_2020.corr(col1, col2)
    print('{0:12s} :: {1:12s} | {2:.5f} | {3:.5f}'.format(
        col1, col2,
        pre, post
    ))
part_b('cold', 'flu')
part_b('cold', 'pneumonia')
part_b('cold', 'coronavirus')
part_b('flu', 'pneumonia')
part_b('flu', 'coronavirus')
part_b('pneumonia', 'coronavirus')
# Big jump in correlation with Covid, smaller jumps (but still quite noticeable) in cross-correlation between other terms

# COMMAND ----------

# EXERCISE 5
# (a)
def is_right_triangle(a, b, c):
    sides = (a**2) + (b**2)
    hypotenuse = c**2
    return sides==hypotenuse

my_is_right_triangle = lambda a, b, c: (c**2) == ((a**2) + (b**2))

# (b)
def clip(x):
    x = max(x, 0)
    x = min(x, 1)
    return x

my_clip = lambda x: min(max(x,0), 1)

def test(x):
    print('{0} | {1} | {2}'.format(
        x,
        clip(x),
        my_clip(x)
    ))
test(1.5)
test(0.5)
test(-0.5)

# COMMAND ----------

# EXERCISE 6
# (a)
def three_values_bigger(row):
    if (row.cold+row.flu+row.pneumonia) > row.coronavirus:
        return 1
    else:
        return 0
print('num days where cold+flu+pneumonia > coronavirus:', 
      post_2020.rdd
      .map(three_values_bigger)
      .reduce(lambda x,y:x+y)
     )
# (b)
print('max difference between cold+flu+pneumonia and coronavirus:',
     post_2020.rdd
      .map(lambda row: (row.cold+row.flu+row.pneumonia)-row.coronavirus)
      .reduce(lambda x,y: max(x,y))
     )
# (c)
row(
    post_2020.rdd
    .map(lambda row: row.coronavirus-(row.cold+row.flu+row.pneumonia))
)
print('mean difference between cold+flu+pneumonia and coronavirus:',
      row_diffs.reduce(lambda x,y: x+y)
      /
      post_2020.count()
     )

# COMMAND ----------

odyssey = spark.read.text("dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/Odyssey.txt")

# COMMAND ----------

# EXERCISE 7
odyssey_words = odyssey.rdd.flatMap(lambda row: row.value.split())
odyssey_words.take(10)

# COMMAND ----------

# EXERCISE 8
odyssey_words = (
    odyssey.rdd
    .flatMap(lambda row: row.value.split())
    .map(lambda word: (word, 1))
    .reduceByKey(lambda x,y: x+y)
    .sortBy(lambda x: x[1], False)
)
odyssey_words.take(25)

# COMMAND ----------

stopwords = spark.read.text('dbfs:/FileStore/shared_uploads/d.r.newman-griffis@sheffield.ac.uk/stopwords.txt')
stopwords = stopwords.collect()
stopwords = set([s['value'] for s in stopwords])
print(stopwords)

# COMMAND ----------

# EXERCISE 9
odyssey_words = (
    odyssey.rdd
    .flatMap(lambda row: row.value.split())
    .map(lambda word: (word, 1))
    .reduceByKey(lambda x,y: x+y)
    .filter(lambda x: x[0] not in stopwords)
    .sortBy(lambda x: x[1], False)
)
odyssey_words.take(25)

# COMMAND ----------


