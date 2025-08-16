import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("heart.csv")

# Features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model as pickle file
with open("heartd.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as heartd.pkl")
