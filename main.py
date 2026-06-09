import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the data
train_df = pd.read_csv('calcium_train_data.csv')
test_df = pd.read_csv('calcium_test_data.csv')
check_df = pd.read_csv('calcium_check_data.csv')


# 2. Separate Features (X) and Targets (y)
X_train = train_df[['Sensor_1', 'Sensor_2']]
y_train = train_df['Target_Calcium']

X_test = test_df[['Sensor_1', 'Sensor_2']]
y_test = test_df['Target_Calcium']

X_check = check_df[['Sensor_1', 'Sensor_2']]
y_check = check_df['Target_Calcium']

# 3. Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Test the model's accuracy on the Test dataset
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# 5. Final validation on the 'Checking' dataset
check_predictions = model.predict(X_check)
check_accuracy = accuracy_score(y_check, check_predictions)
print(f"Checking (Validation) Accuracy: {check_accuracy * 100:.2f}%")