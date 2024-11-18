import pyspark
from delta import *
from delta.tables import *
from pyspark.sql.functions import *

# Courtesy of: https://docs.delta.io/latest/quick-start.html#prerequisite-set-up-java&language-python

# PART ONE
# SETTING UP AND CREATING A BASIC TABLE 

# Setup Python Project
builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Create a table
data = spark.range(0, 10)
data.write.format("delta").mode("overwrite").save("/tmp/delta-table") # Overwrite the previous table

# Read data
df = spark.read.format("delta").load("/tmp/delta-table")
df.show()


# ==================================================
# PART TWO 
# PERFORMING OPERATIONS ON THE TABLE

deltaTable = DeltaTable.forPath(spark, "/tmp/delta-table")

# Update every odd value by adding 1000 to it
deltaTable.update(
  condition = expr("id % 2 != 0"),
  set = { "id": expr("id + 1000") })

# Delete every even value
deltaTable.delete(condition = expr("id % 2 == 0"))

# Create new data to merge with old data
newData = spark.range(0, 20)

deltaTable.alias("oldData") \
  .merge(
    newData.alias("newData"),
    "oldData.id = newData.id") \
  .whenMatchedUpdate(set = { "id": col("newData.id") }) \
  .whenNotMatchedInsert(values = { "id": col("newData.id") }) \
  .execute()

deltaTable.toDF().show()

# ==================================================
# PART THREE 
# PERFORMING TIME TRAVELS
df = spark.read.format("delta").option("versionAsOf", 0).load("/tmp/delta-table")
df.show()

# ==================================================
# PART FOUR 
# Write a stream of data to a table
streamingDf = spark.readStream.format("rate").load()
stream = streamingDf.selectExpr("value as id").writeStream.format("delta").option("checkpointLocation", "/tmp/checkpoint").start("/tmp/delta-table")
stream.stop()

# ==================================================
# PART FIVE 
# Read a stream of changes from a table
stream2 = spark.readStream.format("delta").load("/tmp/delta-table").writeStream.format("console").start()
stream2.stop()