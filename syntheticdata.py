import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Define the dataset (4 samples total, meeting the minimum of 2)
# These features could represent arbitrary metrics, such as sensor readings 
# evaluating the presence of calcium ions (Target: 1 = present, 0 = absent).
X = np.array([
    [7.4, 1.2],  # Sample 1
    [6.1, 0.4],  # Sample 2
    [8.0, 1.5],  # Sample 3
    [5.5, 0.2]   # Sample 4
])

y = np.array([1, 0, 1, 0])

# 2. Initialize and train the algorithm
model = LogisticRegression()
model.fit(X, y)

# 3. Make a prediction on new data
new_sample = np.array([[7.0, 1.0]])
prediction = model.predict(new_sample)
probability = model.predict_proba(new_sample)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")