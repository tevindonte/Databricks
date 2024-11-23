# Databricks notebook source
# MAGIC %md
# MAGIC Analytics

# COMMAND ----------

import matplotlib 
import sklearn
import pandas as pd
import mlflow
import numpy as np

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/databricks-datasets/nyctaxi"))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SHOW CREATE TABLE samples.nyctaxi.trips

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select * from samples.nyctaxi.trips

# COMMAND ----------

 display(dbutils.fs.ls("dbfs:/FileStore/tables/tesla_data_full.csv"))


# COMMAND ----------

df = spark.sql("select * from samples.nyctaxi.trips")
pandas_df = df.toPandas()
print(pandas_df)

# COMMAND ----------

pandas_df

# COMMAND ----------

#How many trips started and ended in the same zip code - Done
#Average trip fair amount + Average Distance - Done
#Trip Distance vs Fare Amount - Done
#Time of day analysis (Morning, Afternoon, Evening, Night) - DOne
#Time of day where trips are most common - Done
#Average costs during those times of day

# COMMAND ----------

avtrip = pandas_df['trip_distance'].mean()
avfare = pandas_df['fare_amount'].mean()
print("The average trip distance is:", round(avtrip,2), "miles", "and the average fare amount is: $", round(avfare,2))


# COMMAND ----------

start = pandas_df['pickup_zip']
end = pandas_df['dropoff_zip']

totalcount = len(pandas_df)
count = 0

for i in range(totalcount):
    if start[i] == end[i]:
        count += 1

startend = count/totalcount * 100
print("The percentage of trips started and ended in the same zip code is:", round(startend,2), "%")

# COMMAND ----------

import statsmodels.api as sm

X = pandas_df['trip_distance']
Y = pandas_df['fare_amount']

X = sm.add_constant(X)
model = sm.OLS(Y,X).fit()
print(model.summary())

# COMMAND ----------

pandas_df['tpep_pickup_datetime'] = pd.to_datetime(pandas_df['tpep_pickup_datetime'])
pandas_df['tpep_dropoff_datetime'] = pd.to_datetime(pandas_df['tpep_dropoff_datetime'])

def categorize(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
    

pandas_df['time_of_day'] = pandas_df['tpep_pickup_datetime'].dt.hour.apply(categorize)
time_of_day_counts = pandas_df['time_of_day'].value_counts()
print(time_of_day_counts)

avg_fare_by_time = pandas_df.groupby('time_of_day')['fare_amount'].mean()
print(avg_fare_by_time)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
sns.countplot(data=pandas_df,x='time_of_day',order=['Morning','Afternoon','Evening','Night'])
plt.xlabel('Time of Day')
plt.ylabel('Number of Trips')
plt.title('Number of Trips by Time of Day')
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x=avg_fare_by_time.index,y=avg_fare_by_time.values, order=['Morning','Afternoon','Evening','Night'])
plt.xlabel('Time of Day')
plt.ylabel ('Average Fare')
plt.title('Average Fare by Time of Day')
plt.show()

# COMMAND ----------

from scipy.stats import ttest_ind

morning_fare = pandas_df[pandas_df['time_of_day'] == 'Morning']['fare_amount']
afternoon_fare = pandas_df[pandas_df['time_of_day'] == 'Afternoon']['fare_amount']

t_test, p_value = ttest_ind(morning_fare, afternoon_fare)
print(f'T-test: {t_test}, P-value: {p_value}')

if p_value < 0.05:
    print("There is a significant difference in the average fare between morning and afternoon trips.")
else:
    print("There is no significant difference in the average fare between morning and afternoon trips.")

# COMMAND ----------

from scipy.stats import chi2_contingency
def categorize_trip_distance(distance):
    if distance < 2:
        return 'Short'
    if 2 <= distance < 5:
        return 'Medium'
    else:
        return 'Long'
    
pandas_df['trip_distance_category'] = pandas_df['trip_distance'].apply(categorize_trip_distance)
contigency_table = pd.crosstab(pandas_df['time_of_day'], pandas_df['trip_distance_category'])
chi2_stat, p_value, dof, expected = chi2_contingency(contigency_table)

print(f'Chi-squared statistic: {chi2_stat}')
print(f'P-value: {p_value}')

if p_value < 0.05:
    print("There is a significant difference between the time of day and the trip distance category.")
else:
    print("There is no significant difference between the time of day and the trip distance category.")

# COMMAND ----------



# COMMAND ----------

