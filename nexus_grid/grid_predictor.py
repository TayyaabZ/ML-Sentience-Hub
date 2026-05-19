from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    spark = (
        SparkSession.builder
        .appName("Nexus-Grid-Pipeline")
        .getOrCreate()
    )

    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("../data/nexus_raw_telemetry.csv")
    )

    cleaned_df = raw_df.filter(raw_df.timestamp != "CORRUPTED_TIME")

    train_df, test_df = cleaned_df.randomSplit([0.8, 0.2], seed=42)

    sector_indexer = StringIndexer(
        inputCol="sector_id",
        outputCol="sector_id_idx",
        handleInvalid="keep",
    )

    status_indexer = StringIndexer(
        inputCol="grid_status",
        outputCol="grid_status_idx",
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=[
            "kw_draw",
            "temperature_c",
            "voltage_drop",
            "sector_id_idx",
        ],
        outputCol="features",
    )

    classifier = GBTClassifier(
        featuresCol="features",
        labelCol="grid_status_idx",
        maxIter=20,
        seed=42,
    )

    pipeline = Pipeline(stages=[
        sector_indexer,
        status_indexer,
        assembler,
        classifier,
    ])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="grid_status_idx",
        predictionCol="prediction",
    )

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    # Save the model
    model.write().overwrite().save("../models/nexus_model")
    spark.stop()


if __name__ == "__main__":
    main()
