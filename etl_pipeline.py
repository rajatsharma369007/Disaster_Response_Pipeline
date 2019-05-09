'''
This script is part of Udacity's Disaster Response project, ETL Pipeline 
preparation
'''
# importing libraries
import pandas as pd

# reading dataset
messages = pd.read_csv("./dataset/messages.csv")
print(messages.head())
categories = pd.read_csv("./dataset/categories.csv")
print(categories.head())

# merging the two dataframe
df = pd.merge(messages, categories, on='id')
print(df.head)

# split categories into separate category columns
categories = df.categories()

