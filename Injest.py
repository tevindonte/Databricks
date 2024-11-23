# Databricks notebook source

import requests
import json

api_key = '5a23f97db239ef07dcfc8e40d8ed628b'
endpoint = 'http://api.aviationstack.com/v1/flights'
params = {
    'access_key': api_key,
    'dep_iata': 'JFK',
    'flight_status': 'active'
}
response = requests.get(endpoint, params=params)
if response.status_code == 200:
    flight_data = response.json()
    print(json.dumps(flight_data, indent=4))
else:
    print(f"Error: {response.status_code}")

# COMMAND ----------

import pandas as pd 

if response.status_code == 200:
    flight_data = response.json()

    flights = flight_data.get('data', [])
    df = pd.json_normalize(flights)
    display(df)
else:
    print(f"Error: {response.status_code}")

# COMMAND ----------

df.to_csv('/dbfs/tmp/sample.csv')

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tmp"))

# COMMAND ----------

# Read the CSV file into a Spark DataFrame
df = spark.read.csv("dbfs:/tmp/sample.csv", header=True, inferSchema=True)

# Create a temporary view from the DataFrame
df.createOrReplaceTempView("sample_view")

# Perform a SELECT statement using Spark SQL
result_df = spark.sql("SELECT * FROM sample_view")

# Save the DataFrame as a table in your organization's data catalog
result_df.write.mode("overwrite").saveAsTable("practice.default.flight")

# Display the result
display(result_df)