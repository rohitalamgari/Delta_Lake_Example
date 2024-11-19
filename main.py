import pyspark
from delta import *
from delta.tables import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression

builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Load Training Data
csv_path = "train.csv"
csv_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(csv_path)
delta_path = "/tmp/delta-csv-table"
csv_df.write.format("delta").mode("overwrite").save(delta_path)

# Get the data we need, clean it up too
delta_df = spark.read.format("delta").load(delta_path)
selected_columns = ["Pclass", "Sex", "Age", "Fare", "Survived"]
titanic_df = delta_df.select(selected_columns)
titanic_df = titanic_df.fillna({"Age": titanic_df.selectExpr("mean(Age)").first()[0],
                                "Fare": titanic_df.selectExpr("mean(Fare)").first()[0]})
titanic_df = titanic_df.withColumn("Sex", when(col("Sex") == "male", 1).otherwise(0))
titanic_df.show(10)
# Columns we care about
feature_columns = ["Pclass", "Sex", "Age", "Fare"]
target_column = "Survived"

# Assemble features into a single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
ml_ready_data = assembler.transform(titanic_df).select("features", target_column)
ml_ready_data.show(10)

# Train the Model
lr = LogisticRegression(featuresCol="features", labelCol=target_column)
lr_model = lr.fit(ml_ready_data)
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Run Tests
from pyspark.ml.evaluation import BinaryClassificationEvaluator

test_df = spark.read.option("header", "true").csv("train.csv", inferSchema=True)
indexer = StringIndexer(inputCol="Sex", outputCol="Sex_index")
test_df = indexer.fit(test_df).transform(test_df)
test_df.show(10)

null_columns = [column for column in feature_columns if test_df.filter(col(column).isNull()).count() == test_df.count()]
for column in feature_columns:
    # Check if the column is numeric
    if dict(test_df.dtypes)[column] in ['int', 'double', 'float']:
        mean_value = test_df.select(mean(col(column))).first()[0]
        if mean_value is not None:  # Check if the mean is valid
            test_df = test_df.fillna({column: mean_value})
        else:
            print(f"Skipping {column} as it doesn't have a valid mean value.")
    else:
        print(f"Skipping non-numeric column: {column}")

# Now update feature_columns to include the indexed 'Sex_index' instead of 'Sex'
feature_columns = [col for col in feature_columns if col != 'Sex'] + ['Sex_index']

# Apply VectorAssembler to create the "features" column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
test_data = assembler.transform(test_df)
test_data.show(10)
# Check the transformed columns
test_data.select("features", *feature_columns).show(10)

# Select the required columns for prediction (features and target_column)
if target_column not in test_data.columns:
    print(f"Target column {target_column} is missing. Available columns: {test_data.columns}")
    target_column = "Survived_index"

test_data = test_data.select("features", target_column)

# Make predictions on the test data
predictions = lr_model.transform(test_data)

# Show predictions
predictions.select("features", target_column, "prediction").show(10)

# Evaluate the model's performance using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol=target_column)
accuracy = evaluator.evaluate(predictions)

print(f"Model Accuracy: {accuracy}")