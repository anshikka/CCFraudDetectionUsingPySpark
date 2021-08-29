# Databricks notebook source
# MAGIC %md
# MAGIC # Credit Card Fraud Detection using PySpark + MLLib
# MAGIC 
# MAGIC ## Using supervised learning, we will explore a dataset to detect a sample fraudulent credit card transactions.
# MAGIC 
# MAGIC ### Meta
# MAGIC Name: Ansh Sikka
# MAGIC 
# MAGIC Dataset Source: Kaggle
# MAGIC 
# MAGIC Dataset Link: (https://www.kaggle.com/ealaxi/paysim1)
# MAGIC 
# MAGIC Date: 08/29/2021

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC curl -O 'https://fraud-detection-ansh-sikka.s3.us-east-2.amazonaws.com/fraud_transactions.csv'

# COMMAND ----------

# MAGIC %fs ls "file:/databricks/driver"

# COMMAND ----------

import pandas as pd
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

# COMMAND ----------

# Load data
# define path to file
path = 'file:/databricks/driver/fraud_transactions.csv'

# load data using sqlContext
df = sqlContext.read.format("csv")\
      .option("header", "true")\
      .option("inferSchema", "true")\
      .load(path)\
      .limit(5000)

# display in table format


# COMMAND ----------

# Show number of flagged transactions
df.filter(df.isFlaggedFraud == 1).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### As we can see above, none of these transactions were flagged as fraud! Let's see how we can explore different supervised and unsupervised algorithms to detect these anomalies in credit card transactions.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Now that we have the data loaded, let's take grab the column names that hold catigorical data and put them into a list.

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder

# stages in pipeline
stages = []
numericColumns = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "newbalanceDest"]
categoricalColumns = ["type"]

for categoricalCol in categoricalColumns:
  stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
  encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
  stages+=[stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol="isFraud", outputCol="label")
stages+=[label_stringIdx]



# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Let's now put all the feature columns into a single vector columns

# COMMAND ----------

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericColumns
assembler = VectorAssembler(inputCols = assemblerInputs, outputCol="features")
stages+=[assembler]

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Let's now run the stages as a pipeline. We can put all the feature transformations under a single call.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
cols = df.columns

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(df)
preppedDataDF = pipelineModel.transform(df)
preppedDataDF.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Let's extract the columns that are needed for the training and testing.

# COMMAND ----------

# take the vector columns and the original columns
selectedcols = ["label", "features"] + cols
dataset = preppedDataDF.select(selectedcols)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Split the training and testing data: 30% Testing, 70% Training

# COMMAND ----------

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print("Training data points: " + str(trainingData.count()))
print("Testing data points: " + str(testData.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Let's use logistic regression. This algorithm will have an output of 0 or 1 (Great for Binary Classification)

# COMMAND ----------

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
 
# Train model with Training Data
lrModel = lr.fit(trainingData)
# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)
# View model's predictions and probabilities of each prediction class
# You can select any columns in the above schema to view as well
selected = predictions.select("label", "prediction", "probability", "type", "amount")

display(selected.filter(selected.label==1))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Here's a start. At first, none of them were flagged as fraud. Even though the amount we flagged as fraud is low, we still have a better chance at detections now!

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# Create both evaluators
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')

predictionAndTarget = predictions.select("label", "prediction")

acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

print("Accuracy: " + str(acc))
print("F1 Score: " + str(f1))
print("Weighted Precision: " + str(weightedPrecision))
print("Weighted Recall: " + str(weightedRecall))
print("Area Under Curve: " + str(auc))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Woah! Why are these metrics so high: Here's an explanation (from Spark MLLib Documentation):
# MAGIC While there are many different types of classification algorithms, the evaluation of classification models all share similar principles. In a supervised classification problem, there exists a true output and a model-generated predicted output for each data point. For this reason, the results for each data point can be assigned to one of four categories:
# MAGIC 
# MAGIC True Positive (TP) - label is positive and prediction is also positive
# MAGIC True Negative (TN) - label is negative and prediction is also negative
# MAGIC False Positive (FP) - label is negative but prediction is positive
# MAGIC False Negative (FN) - label is positive but prediction is negative
# MAGIC These four numbers are the building blocks for most classifier evaluation metrics. A fundamental point when considering classifier evaluation is that pure accuracy (i.e. was the prediction correct or incorrect) is not generally a good metric. The reason for this is because a dataset may be highly unbalanced. For example, if a model is designed to predict fraud from a dataset where 95% of the data points are not fraud and 5% of the data points are fraud, then a naive classifier that predicts not fraud, regardless of input, will be 95% accurate. For this reason, metrics like precision and recall are typically used because they take into account the type of error. In most applications there is some desired balance between precision and recall, which can be captured by combining the two into a single metric, called the F-measure.
# MAGIC 
# MAGIC However, we should take a look at the Area Under Curve (AUC). It's a value that is 0.5 < AUC < 1. The closer it is to 0.5, the less the classifier is able to distinguish between fraud and not fraud. Since 0.56 is closer to 0.5 than 1, we can see the model isn't good at all for determining fraud.

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
 
# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=10)
 
# Train model with Training Data
dtModel = dt.fit(trainingData)
predictions = dtModel.transform(testData)
selected = predictions.select("label", "prediction", "probability", "type", "amount")
display(selected.filter(selected.label==1))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### We can see that there are more accurate predictions this time around! Let's see if we can get a better model though.

# COMMAND ----------

# Extract Results
predictionAndTarget = predictions.select("label", "prediction")

acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

print("Accuracy: " + str(acc))
print("F1 Score: " + str(f1))
print("Weighted Precision: " + str(weightedPrecision))
print("Weighted Recall: " + str(weightedRecall))
print("Area Under Curve: " + str(auc))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Woah! Why are these metrics so high: Here's an explanation (from Spark MLLib Documentation):
# MAGIC While there are many different types of classification algorithms, the evaluation of classification models all share similar principles. In a supervised classification problem, there exists a true output and a model-generated predicted output for each data point. For this reason, the results for each data point can be assigned to one of four categories:
# MAGIC 
# MAGIC True Positive (TP) - label is positive and prediction is also positive
# MAGIC True Negative (TN) - label is negative and prediction is also negative
# MAGIC False Positive (FP) - label is negative but prediction is positive
# MAGIC False Negative (FN) - label is positive but prediction is negative
# MAGIC These four numbers are the building blocks for most classifier evaluation metrics. A fundamental point when considering classifier evaluation is that pure accuracy (i.e. was the prediction correct or incorrect) is not generally a good metric. The reason for this is because a dataset may be highly unbalanced. For example, if a model is designed to predict fraud from a dataset where 95% of the data points are not fraud and 5% of the data points are fraud, then a naive classifier that predicts not fraud, regardless of input, will be 95% accurate. For this reason, metrics like precision and recall are typically used because they take into account the type of error. In most applications there is some desired balance between precision and recall, which can be captured by combining the two into a single metric, called the F-measure.
# MAGIC 
# MAGIC However, we should take a look at the Area Under Curve (AUC). It's a value that is 0.5 < AUC < 1. The closer it is to 0.5, the less the classifier is able to distinguish between fraud and not fraud. Since 0.66 is closer to 0.5 than 1, we can see the model isn't that great (but better than logisitic regression) for determining fraud.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC For the last part, let's try a Random Forest Classifier, which use an ensemble (group) of decision trees to improve accuracy. 

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
 
# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")
 
# Train model with Training Data
rfModel = rf.fit(trainingData)
# Make predictions on test data using the Transformer.transform() method.
predictions = rfModel.transform(testData)

# COMMAND ----------

selected = predictions.select("label", "prediction", "probability", "type", "amount")

display(selected.filter(selected.label==1))

# COMMAND ----------

# Extract Results
predictionAndTarget = predictions.select("label", "prediction")

acc = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "accuracy"})
f1 = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "f1"})
weightedPrecision = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedPrecision"})
weightedRecall = evaluatorMulti.evaluate(predictionAndTarget, {evaluatorMulti.metricName: "weightedRecall"})
auc = evaluator.evaluate(predictionAndTarget)

print("Accuracy: " + str(acc))
print("F1 Score: " + str(f1))
print("Weighted Precision: " + str(weightedPrecision))
print("Weighted Recall: " + str(weightedRecall))
print("Area Under Curve: " + str(auc))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### This seems like the lowest AOC. The random forest classifier wasn't very successful.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Now that we looked at some unsupervised learning algorithms, we can conclude that based on our selected features and our data, a decision tree was the best performing algorithm, albiet not performing as well either. However, the original dataset stated that none of the transactions were originally identified as fraud, so finding some fraudulent transactions showed that we made some progress!

# COMMAND ----------

# MAGIC %md
# MAGIC # Future Considerations
# MAGIC 
# MAGIC ## Unsupervised Learning
# MAGIC 
# MAGIC ### With unsupervised learning, we won't be using any labels. The algorithm will be analyzing similar datapoints and detecting possible outliers.
# MAGIC ### We could try using K-Means Clustering, but we want to detect anomalies, so we should start with a 1-class support vector machine (SVM). Since the data has a low distribution of fraudulent transactions, this would be the best option. Unfortunately, PySpark and MLLib don't have a package for 1-Class SVMs. 
# MAGIC 
# MAGIC ## Plotting
# MAGIC ### We could possibly plot the data to further adjust the hyperparameters in the model.
# MAGIC 
# MAGIC ## Hyperparameter Tuning and Evaluation
# MAGIC ### We could possibly test each algoirthm using different hyperparameters and chosing the combination of parameters + algorithm type to determine the best outcome
