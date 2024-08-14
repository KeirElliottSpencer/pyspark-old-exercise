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
patient_info_provinces = patient_info.select('province').distinct()
weather_provinces = weather.select('province').distinct()

# (a)
print('Patient info: {0:,} provinces'.format(patient_info_provinces.count()))
print('Weather: {0:,} provinces'.format(weather_provinces.count()))
print('Intersection: {0:,} provinces'.format(patient_info_provinces.intersect(weather_provinces).count()))

# (b)
print('Difference :: Patient info only -- {0} || Weather only -- {1}'.format(
    patient_info_provinces.subtract(weather_provinces).collect(), 
    weather_provinces.subtract(patient_info_provinces).collect()
))

# One province in each that does not appear in the other. Will need to be conscious of this when joining but otherwise this will likely not have much impact.

# COMMAND ----------

# EXERCISE 2
(
    patient_info.rdd
    .map(lambda r: r.confirmed_date)
    .reduce(min)                         # min is a pre-built function in Python that takes two (or more) inputs, so can use it directly here
)
# This fails! There is a None value somewhere in the confirmed_date column...

# COMMAND ----------

# EXERCISE 3
# (a)
def countAndDropNa(tbl, col, lbl):
    tbl_dates = tbl.select(col)
    tbl_pre_drop = tbl_dates.count()
    tbl_dates = tbl_dates.dropna()
    tbl_post_drop = tbl_dates.count()
    
    print('Table {0} --  Pre-drop: {1:,} rows  Post-drop: {2:,} rows   Null values filtered: {3:,} rows'.format(
        lbl,
        tbl_pre_drop,
        tbl_post_drop,
        tbl_pre_drop-tbl_post_drop
    ))
    
    return tbl_dates

patient_info_dates = countAndDropNa(patient_info, 'confirmed_date', 'PatientInfo')
weather_dates = countAndDropNa(weather, 'date', 'Weather')
search_trend_dates = countAndDropNa(search_trend, 'date', 'SearchTrend')

# Three rows in PatientInfo have null confirmed_date. No null dates in Weather or SearchTrend.

# (b)
def getDateRange(dates, col, lbl):
    dates = dates.rdd.map(lambda r: r[col])
    print('Table {0} date range --  Min: {1}  Max: {2}'.format(
        lbl,
        dates.reduce(min),
        dates.reduce(max)
    ))
getDateRange(patient_info_dates, 'confirmed_date', 'PatientInfo')
getDateRange(weather_dates, 'date', 'Weather')
getDateRange(search_trend_dates, 'date', 'SearchTrend')

# Generally well-aligned: lots of extradata in Weather and SearchTrend, but only 1 day (30th June) that we have PatientInfo data for but not data from Weather or SearchTrend.

# COMMAND ----------

# EXERCISE 4
# Start with fixing data types in SearchTrend
search_trend = (search_trend
                .withColumn('cold', search_trend.cold.cast('double'))
                .withColumn('flu', search_trend.flu.cast('double'))
                .withColumn('pneumonia', search_trend.pneumonia.cast('double'))
                .withColumn('coronavirus', search_trend.coronavirus.cast('double'))
               )

# (a)
high_cold = search_trend.where(search_trend.cold >= 1)
high_covid = search_trend.where(search_trend.coronavirus >= 1)

def countIt(tbl,lbl):
    print('{0} --> {1:,} rows'.format(lbl, tbl.count()))
countIt(high_cold, 'high_cold')
countIt(high_covid, 'high_covid')
countIt(high_cold.union(high_covid), 'either trending (union)')
countIt(high_cold.intersect(high_covid), 'both trending (intersection)')
countIt(high_cold.subtract(high_covid), 'cold but not coronavirus (subtract)')

# (b)
getDateRange(high_cold, 'date', 'high_cold')
getDateRange(high_covid, 'date', 'high_covid')
getDateRange(high_cold.intersect(high_covid), 'date', 'cold/covid intersect')
# Intersection is as expected: in 2020. Surprising to see coronavirus spike in 2018! (An earlier coronavirus.)

# (c)
print('High cold and flu')
high_cold.where(high_cold.flu >= 1).show()
print ('High cold and pneumonia')
high_cold.where(high_cold.pneumonia >= 1).show()
print ('High covid and flu')
high_covid.where(high_covid.flu >= 1).show()
print ('High covid and pneumonia')
high_covid.where(high_covid.pneumonia >= 1).show()
# Cold+flu is surprising! Must have been a bad December. All other high overlaps are in 2020, which makes sense.

# COMMAND ----------

# EXERCISE 5

# Start with fixing the avg_temp data type in Weather
weather = (
    weather
    .withColumn('avg_temp', weather.avg_temp.cast('double'))
)

# Get 2020 data only
search_trend_2020 = search_trend.where(search_trend.date >= '2020-01-01')
weather_2020 = weather.where(weather.date >= '2020-01-01')

# Drop null values from the columns we're interested in
search_trend_clean = search_trend_2020.dropna(subset='coronavirus')
weather_clean = weather_2020.dropna(subset='avg_temp')

# Map (in the RDD API) to the values we want to get from each table
covid_searches_by_date = (
    search_trend_clean.rdd
    .map(lambda r: (r.date, r.coronavirus))
)
weather_temps_by_date = (
    weather_clean.rdd
    .map(lambda r: (r.date, r.avg_temp))
)

# Check how many different values we have for each date in Weather: it's 16, for the 16 provinces!
weather_temps_by_date.countByKey()

# Group the provincial average temperatures by date, and average across all provinces to get the overall average daily temperature
weather_temps_by_date = weather_temps_by_date.groupByKey()
weather_temps_by_date = (
    weather_temps_by_date
    .map(lambda pair: (
        pair[0],
        (sum(pair[1])/len(pair[1]))
    ))
)

# Join the two data sources (both mapped to date keys)
date_info = covid_searches_by_date.join(weather_temps_by_date)

# Pop out of the nested tuple structure and map to a DataFrame
date_info = date_info.map(lambda r: (r[0], r[1][0], r[1][1]))
date_info = date_info.toDF(['date', 'search', 'avg_temp'])

# Finally, calculate our correlation!
date_info.corr('search', 'avg_temp')

# This supports our hypothesis! This is a reasonably strong correlation, actually, given the complexity of infectious diseases.

# COMMAND ----------

# EXERCISE 6

# Drop null values in PatientInfo's confirmed_date column
patient_info_dated = patient_info.dropna(subset='confirmed_date')

# Map each date to patient IDs...
patients_by_date = patient_info_dated.rdd.map(
    lambda r: (r.confirmed_date, r.patient_id)
)
# And aggregate those IDs by grouping and counting
patients_by_date = patients_by_date.groupByKey()
patients_by_date = patients_by_date.map(lambda pair: (pair[0], len(pair[1])))

# Use join(), just as we did before
date_info = patients_by_date.join(covid_searches_by_date)
# Pop out of the nested structure and cast to a DataFrame, just as we did before
date_info = date_info.map(lambda r: (r[0], r[1][0], r[1][1]))
date_info = date_info.toDF(['date', 'num_patients', 'search'])

# Calculate our correlation!
date_info.corr('num_patients', 'search')

# This correlation also supports our hypothesis, although it is fairly weak.

# COMMAND ----------

# EXERCISE 7
def getYearAndMonth(date):
    (year, month, day) = date.split('-')
    return '-'.join([year, month])

# (a)
# Get rid of records with null value for confirmed_date
patient_info_dated = patient_info.dropna(subset='confirmed_date')

# Map patients by province and month
patients_by_province_and_month = (
    patient_info_dated.rdd.map(
        lambda r: (r.province, getYearAndMonth(r.confirmed_date), r.patient_id)
    )
)

# Convert back to a DataFrame...
patients_by_province_and_month = patients_by_province_and_month.toDF(['province', 'month', 'patient_id'])
# ...use groupBy() to group data by province and month...
patients_by_province_and_month = patients_by_province_and_month.groupBy(['province', 'month'])
# ...and use count() to aggregate back to a DataFrame.
num_patients_by_province_and_month = patients_by_province_and_month.count()  # Returns a DataFrame with 'count' column


# (b)
# Map temperatures by province and month
temps_by_province_and_month = (
    weather.rdd.map(
        lambda r: (r.province, getYearAndMonth(r.date), r.avg_temp)
    )
)

# Convert to a DataFrame, group, and average
temps_by_province_and_month = temps_by_province_and_month.toDF(['province', 'month', 'avg_temp'])
temps_by_province_and_month = temps_by_province_and_month.groupBy(['province', 'month'])
temps_by_province_and_month = temps_by_province_and_month.avg('avg_temp')
temps_by_province_and_month = temps_by_province_and_month.withColumnRenamed('avg(avg_temp)', 'avg_temp')

# (c)
# Use DataFrame conditional-based join to handle multiple keys
combo = num_patients_by_province_and_month.join(
    temps_by_province_and_month,
    (
        (num_patients_by_province_and_month.province == temps_by_province_and_month.province)
        &
        (num_patients_by_province_and_month.month == temps_by_province_and_month.month)
    )
)
# And calculate correlation
combo.corr('count', 'avg_temp')

# This does NOT support our hypothesis!
