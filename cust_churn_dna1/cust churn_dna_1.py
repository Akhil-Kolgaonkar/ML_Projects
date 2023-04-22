import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the data from a CSV file
df = pd.read_csv('customer_data.csv')

# Clean and preprocess the data
df = df.dropna()  # Remove rows with missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # Convert TotalCharges to numeric values
df = df.dropna()  # Remove rows with missing TotalCharges values

# Calculate basic statistics and visualizations
num_customers = len(df)
churn_count = df['Churn'].value_counts()
plt.pie(churn_count.values, labels=churn_count.index, autopct='%1.1f%%')
plt.title('Churn Distribution')
plt.show()

tenure_hist = df['tenure'].hist()
tenure_hist.set_xlabel('Tenure (months)')
tenure_hist.set_ylabel('Count')
tenure_hist.set_title('Tenure Distribution')
plt.show()

# Prepare the data for modeling
X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
X = pd.get_dummies(X)  # One-hot encode categorical features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split into training and test sets

# Train a logistic regression model to predict churn
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Evaluate the model's performance on the test set
y_pred = logreg.predict(X_test)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Identify important features in the model
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': logreg.coef_[0]})
feature_importances = feature_importances.sort_values('importance', ascending=False)
print('Most important features:\n', feature_importances.head())

# Make recommendations for improving customer retention based on the analysis
