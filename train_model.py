import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Create models folder
os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

# Encode Gender
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Remove CustomerID
df = df.drop("CustomerID", axis=1)

# Scale Data
scaler = MinMaxScaler()
X = scaler.fit_transform(df)

# Encoder
input_layer = Input(shape=(4,))

encoded = Dense(8, activation="relu")(input_layer)
encoded = Dense(2, activation="relu")(encoded)

# Decoder
decoded = Dense(8, activation="relu")(encoded)
decoded = Dense(4, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.fit(
    X,
    X,
    epochs=100,
    batch_size=16,
    validation_split=0.2
)

autoencoder.save("models/autoencoder.keras")

print("Model Saved Successfully")