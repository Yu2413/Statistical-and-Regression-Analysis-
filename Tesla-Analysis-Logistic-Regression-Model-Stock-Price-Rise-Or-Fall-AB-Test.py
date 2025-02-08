# Import necessary libraries
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Collection
# Download Tesla stock data
tesla_data = yf.download("TSLA", start="2020-01-01", end="2025-02-07")

# Create the target variable: 1 if the stock price falls the next day, 0 otherwise
tesla_data['Target'] = (tesla_data['Close'].shift(-1) < tesla_data['Close']).astype(int)
tesla_data.dropna(inplace=True)

# Step 2: Feature Engineering
# Add a simple moving average as a feature
tesla_data['SMA_50'] = tesla_data['Close'].rolling(window=50).mean()
tesla_data.dropna(inplace=True)

# Define features and target
X = tesla_data[['Close', 'SMA_50']]  # Add more features as needed
y = tesla_data['Target']

# Step 3: Data Splitting
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Model Interpretation
# Display the coefficients of the model
coefficients = pd.DataFrame(model.coef_.T, X.columns, columns=['Coefficient'])
print("Model Coefficients:\n", coefficients)

# Step 7: Prediction
# Example new data for prediction
new_data = pd.DataFrame({'Close': [250], 'SMA_50': [245]})  # Replace with actual values
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Will Tesla's stock fall tomorrow?", "Yes" if prediction[0] == 1 else "No")