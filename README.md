# -SCT_DS_4-

#Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("population_dataset.csv")
df

#Basic EDA
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe(include='all'))


#Visualizations
#Age Distribution (Histogram)
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Gender Distribution (Bar Chart)
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Education Distribution (Bar Chart)
plt.figure(figsize=(8,4))
sns.countplot(x='Education', data=df, order=df['Education'].value_counts().index, palette='Set3')
plt.title("Education Distribution")
plt.xlabel("Education")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Age & Income)")
plt.tight_layout()
plt.show()

#Boxplot of Income by Education
plt.figure(figsize=(8,4))
sns.boxplot(x='Education', y='Income', data=df, palette='Set1')
plt.title("Income by Education Level")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


