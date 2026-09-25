# 🛒 SmartCart — Customer Segmentation using Machine Learning

SmartCart is a machine learning-based customer segmentation project that analyzes customer demographics, income, household characteristics, and spending behavior to identify meaningful customer groups. The project uses unsupervised learning techniques such as **K-Means and Agglomerative Clustering**, along with **PCA, Elbow Method, and Silhouette Score**, to discover and visualize customer segments.

## 🎯 Problem Statement

Businesses have customers with different purchasing patterns, income levels, demographics, and household characteristics. Treating every customer in the same way can make it difficult to understand their behavior and create targeted strategies.

SmartCart aims to segment customers into meaningful groups based on their characteristics and spending behavior, helping generate useful insights for targeted marketing and data-driven decision-making.

## 🚀 Key Features

- Customer data preprocessing and cleaning
- Missing-value handling
- Feature engineering
- Customer age and tenure calculation
- Total spending analysis
- One-Hot Encoding for categorical features
- Feature scaling using StandardScaler
- Outlier handling
- PCA-based dimensionality reduction
- Elbow Method for cluster selection
- Silhouette Score for cluster evaluation
- K-Means Clustering
- Agglomerative Clustering
- 3D customer segment visualization
- Cluster-wise customer behavior analysis

## 🧠 Machine Learning Workflow

```text
Customer Dataset
       ↓
Data Preprocessing
       ↓
Feature Engineering
       ↓
Encoding & Scaling
       ↓
Outlier Handling
       ↓
PCA
       ↓
Cluster Evaluation
   ↙          ↘
Elbow       Silhouette
Method        Score
   ↓
Clustering
   ↙          ↘
K-Means   Agglomerative
   ↓
Customer Segmentation
       ↓
Cluster Analysis
