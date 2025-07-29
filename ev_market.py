# EV Market Segmentation Based on Age Group
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Step 2: Load Dataset

df = pd.read_csv("Indian automoble buying behavour study 1.0.csv")
df.head()



# Step 3: Age Group Segmentation

age_bins = [20, 30, 40, 50, 60, 70]
age_labels = ['21-30', '31-40', '41-50', '51-60', '61-70']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)


# Step 4: Buyer Distribution by Age Group

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Age Group', palette='viridis')
plt.title("Buyer Distribution by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Buyers")
plt.savefig("buyer_distribution.png")
plt.show()


#Step 5: Average Price Spent per Age Group

plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Age Group', y='Price', estimator=np.mean, ci=None, palette='plasma')
plt.title("Average Vehicle Price per Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Price")
plt.savefig("avg_price_by_age.png")
plt.show()


# Step 6: Most Preferred Make per Age Group

most_common_make = df.groupby('Age Group')['Make'].agg(lambda x: x.value_counts().index[0] if not x.empty else "N/A")
print("\nMost Preferred Make per Age Group:\n")
print(most_common_make)

#  Step 7: Summary Table

age_group_counts = df['Age Group'].value_counts().sort_index()
avg_price_per_age_group = df.groupby('Age Group')['Price'].mean()
summary_df = pd.DataFrame({
    'Buyer Count': age_group_counts,
    'Average Price': avg_price_per_age_group,
    'Most Preferred Make': most_common_make
})
summary_df.reset_index(inplace=True)
summary_df.rename(columns={'index': 'Age Group'}, inplace=True)
summary_df.to_csv("age_group_summary.csv", index=False)
summary_df

# Step 8: KMeans Clustering 

features = df[['Age', 'Total Salary']].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
features['Cluster'] = kmeans.fit_predict(scaled_features)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=features, x='Age', y='Total Salary', hue='Cluster', palette='Set2')
plt.title("Customer Segmentation Using KMeans Clustering")
plt.savefig("kmeans_clusters.png")
plt.show()



