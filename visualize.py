import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("clustered_customers.csv")

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Income")
plt.ylabel("Spending Score")

plt.title("Customer Segmentation")

plt.show()