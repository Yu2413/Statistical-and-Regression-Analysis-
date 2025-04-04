"""
FOREX RECESSION LIKELIHOOD ANALYSIS USING ALPHAVANTAGE

Steps:
1) Fetch Forex data for multiple currency pairs (e.g., EUR/USD, GBP/USD, USD/JPY)
2) Merge and preprocess the data
3) Generate synthetic recession labels (for demonstration)
4) Train a RandomForest model to classify the 'likelihood' of a recession
5) Evaluate and visualize the results

Disclaimer:
- This code is for educational purposes and does not constitute financial advice.
- The recession flag is generated synthetically for illustration.
- For real use, replace it with an actual recession indicator or official data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from alpha_vantage.foreignexchange import ForeignExchange
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


############################################
# 1. HELPER FUNCTIONS
############################################
def fetch_daily_forex_data(api_key, from_symbol, to_symbol, outputsize='compact'):
    """
    Fetch daily forex data from AlphaVantage using the ForeignExchange class.
    Returns:
       df (pd.DataFrame): time series data with columns ['open', 'high', 'low', 'close']
       meta_data (dict): metadata from the AlphaVantage API
    """
    fx = ForeignExchange(key=api_key, output_format='pandas')
    # get_currency_exchange_daily returns (DataFrame, metadata)
    # Possible intervals: get_currency_exchange_intraday(...) for shorter intervals
    df, meta_data = fx.get_currency_exchange_daily(
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        outputsize=outputsize
    )

    # AlphaVantage returns data in descending chronological order (most recent first)
    df.sort_index(inplace=True)

    # Rename columns for clarity
    df.columns = [f'{from_symbol}_{to_symbol}_' + col for col in ['open', 'high', 'low', 'close']]

    return df, meta_data


def generate_synthetic_recession_label(master_df, threshold_change=-0.02):
    """
    Generate a synthetic recession flag for demonstration.
    We'll pretend that if the average close price for all pairs
    has a large negative change, it signals a "recession."
    """
    # Identify all 'close' columns in the master DataFrame
    close_cols = [col for col in master_df.columns if '_close' in col.lower()]

    # Compute the average close across all pairs
    master_df['Forex_Avg_Close'] = master_df[close_cols].mean(axis=1)

    # Compute the percentage change of this average close
    master_df['Forex_Avg_Pct_Change'] = master_df['Forex_Avg_Close'].pct_change().fillna(0)

    # Binary flag if the drop is below a certain threshold (e.g., -2%)
    master_df['Recession_Flag'] = (master_df['Forex_Avg_Pct_Change'] < threshold_change).astype(int)
    return master_df


############################################
# 2. MAIN SCRIPT
############################################
def main():
    # --------------------------------------
    # A) SETUP & FETCH DATA
    # --------------------------------------
    ALPHA_VANTAGE_API_KEY = "TJP9N00T8C5FBBAI"  # <-- replace with your real API key
    currency_pairs = [
        ("EUR", "USD"),
        ("GBP", "USD"),
        ("USD", "JPY")
    ]

    # Dictionary to store DataFrames for each pair
    pair_data = {}

    for (base, quote) in currency_pairs:
        df, meta = fetch_daily_forex_data(
            api_key=ALPHA_VANTAGE_API_KEY,
            from_symbol=base,
            to_symbol=quote,
            outputsize='full'  # 'full' gives the maximum available data
        )
        pair_data[f"{base}_{quote}"] = df

    # --------------------------------------
    # B) MERGE INTO A MASTER DATAFRAME
    # --------------------------------------
    # We'll use outer join on the date index to keep as many data points as possible
    master_df = None
    for _, df in pair_data.items():
        if master_df is None:
            master_df = df.copy()
        else:
            master_df = master_df.join(df, how='outer')

    # Because not all currency pairs might start or end on the exact same date,
    # you can forward-fill or drop missing values. Here, we'll drop for simplicity:
    master_df.dropna(inplace=True)

    # --------------------------------------
    # C) GENERATE SYNTHETIC RECESSION FLAG
    # --------------------------------------
    master_df = generate_synthetic_recession_label(master_df, threshold_change=-0.02)

    # --------------------------------------
    # D) FEATURE ENGINEERING
    # --------------------------------------
    # We'll create daily percentage-change features for each "close" column
    close_cols = [col for col in master_df.columns if '_close' in col.lower()]

    for col in close_cols:
        master_df[f"{col}_pct_change"] = master_df[col].pct_change().fillna(0)

    feature_cols = [f"{col}_pct_change" for col in close_cols]
    target_col = "Recession_Flag"

    # Drop any rows that might have arisen with NaNs after the calculations
    master_df.dropna(subset=feature_cols + [target_col], inplace=True)

    # --------------------------------------
    # E) TRAIN / TEST SPLIT
    # --------------------------------------
    X = master_df[feature_cols]
    y = master_df[target_col]

    # In a real time-series scenario, you might want to do a chronological split
    # (rather than random shuffle), but for demo, we'll do a simple random split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False  # keep chronological order
    )

    # --------------------------------------
    # F) MODEL TRAINING
    # --------------------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --------------------------------------
    # G) EVALUATION
    # --------------------------------------
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    feature_importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:\n", fi_df)

    # --------------------------------------
    # H) VISUALIZATION
    # --------------------------------------
    # 1) Plot the average close and mark recession points
    plt.figure(figsize=(10, 5))
    plt.plot(master_df.index, master_df['Forex_Avg_Close'], label='Forex_Avg_Close')

    # Indicate where Recession_Flag = 1
    recession_indices = master_df.index[master_df['Recession_Flag'] == 1]
    recession_values = master_df['Forex_Avg_Close'][master_df['Recession_Flag'] == 1]
    plt.scatter(recession_indices, recession_values, marker='x', label='Recession Flag')

    plt.title("Forex Average Close with Synthetic Recession Points")
    plt.xlabel("Date")
    plt.ylabel("Avg Close Price")
    plt.legend()
    plt.show()

    # 2) Actual vs Predicted Recession
    plt.figure(figsize=(10, 5))
    plt.plot(master_df.index, master_df[target_col], label='Actual Recession')
    plt.plot(master_df.index[-len(y_test):], y_pred, label='Predicted Recession')
    plt.title("Actual vs. Predicted Recession")
    plt.xlabel("Date")
    plt.ylabel("Recession Flag")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
