import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file
df = pd.read_csv('customer_purchases.csv')

# Print the first few rows of the data
print(df.head())

# Calculate some basic statistics for the age and price fields
print("Age statistics:")
print(df['age'].describe())
print("\nPrice statistics (CAD):")
print(df['price(CAD)'].describe())

# Create a histogram of the age distribution
plt.hist(df['age'], bins=20)
plt.title('Histogram of Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Create a scatterplot of price vs. age
plt.scatter(df['age'], df['price(CAD)'])
plt.title('Price vs. Age')
plt.xlabel('Age')
plt.ylabel('Price (CAD)')
plt.show()

# Create a bar chart of the top 10 professions in the dataset
profession_counts = df['profession'].value_counts().nlargest(10)
plt.bar(profession_counts.index, profession_counts.values)
plt.title('Top 10 Professions')
plt.xlabel('Profession')
plt.ylabel('Count')
plt.show()

# Create a pie chart of the gender distribution
gender_counts = df['gender'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')
plt.show()
