from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def main():
    spark = (
        SparkSession.builder
        .appName("Aegis-Vanguard-Pipeline")
        .config("spark.hadoop.security.authentication", "simple")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )

    raw_df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("../data/aegis_raw_logs.csv")
    )

    train_df, test_df = raw_df.randomSplit([0.8, 0.2], seed=42)

    imputer = Imputer(
        inputCol="session_duration_sec",
        outputCol="session_duration_imputed",
        strategy="median",
    )

    user_agent_indexer = StringIndexer(
        inputCol="user_agent",
        outputCol="user_agent_idx",
        handleInvalid="keep",
    )

    label_indexer = StringIndexer(
        inputCol="class_label",
        outputCol="label_idx",
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=[
            "session_duration_imputed",
            "click_velocity_bps",
            "pages_viewed",
            "user_agent_idx",
        ],
        outputCol="features",
    )

    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_idx",
        numTrees=100,
        maxDepth=10,
        seed=42,
    )

    pipeline = Pipeline(stages=[
        imputer,
        user_agent_indexer,
        label_indexer,
        assembler,
        classifier,
    ])

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_idx",
        predictionCol="prediction",
    )

    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    # Save the model
    model.write().overwrite().save("../models/aegis_model")
    spark.stop()


if __name__ == "__main__":
    main()
