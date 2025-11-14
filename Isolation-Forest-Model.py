from sklearn.ensemble import IsolationForest
import numpy as np

# Sample data
X = np.random.randn(100, 2)  # 100 samples, 2 features

# Train the model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Predict anomalies
anomaly_scores = model.decision_function(X)  # Anomaly scores
predictions = model.predict(X)  # -1 for anomalies, 1 for normal points

# Print results
print("Anomaly Scores:", anomaly_scores)
print("Predictions:", predictions)