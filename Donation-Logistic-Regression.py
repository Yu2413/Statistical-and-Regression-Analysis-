import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Hypothetical dataset
# Features:
# 1. financial_stability: 1 (stable), 0 (unstable)
# 2. cause_importance: 1 (important), 0 (not important)
# 3. emotional_connection: 1 (strong), 0 (weak)
# Target:
# donate: 1 (yes), 0 (no)

data = {
    'financial_stability': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'cause_importance': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    'emotional_connection': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    'donate': [1, 0, 0, 1, 1, 0, 1, 0, 1, 0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['financial_stability', 'cause_importance', 'emotional_connection']]
y = df['donate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to predict whether to donate or not
def should_i_donate(financial_stability, cause_importance, emotional_connection):
    input_data = np.array([[financial_stability, cause_importance, emotional_connection]])
    prediction = model.predict(input_data)
    return "Yes, you should donate!" if prediction[0] == 1 else "No, you should not donate."

# Example usage
print(should_i_donate(1, 1, 1))  # Example: Stable, important cause, strong emotional connection
print(should_i_donate(0, 1, 0))  # Example: Unstable, important cause, weak emotional connection