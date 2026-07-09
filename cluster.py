import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# Load Dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

encoder = LabelEncoder()

df["Gender"] = encoder.fit_transform(df["Gender"])

df = df.drop("CustomerID", axis=1)

scaler = MinMaxScaler()

X = scaler.fit_transform(df)

# Load Model
autoencoder = load_model("models/autoencoder.keras")

# Encoder Part
encoder_model = Model(
    inputs=autoencoder.input,
    outputs=autoencoder.layers[2].output
)

compressed = encoder_model.predict(X)

# KMeans
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

clusters = kmeans.fit_predict(compressed)

df["Cluster"] = clusters

print(df.head())

df.to_csv("clustered_customers.csv", index=False)

print("Cluster file created")