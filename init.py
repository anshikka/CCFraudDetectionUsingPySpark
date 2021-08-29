# preprocessing-numerical
import pandas as pd

#viz
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# spark
import findspark
findspark.init()
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.session import SparkSession

# initialize spark
sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession(sc)