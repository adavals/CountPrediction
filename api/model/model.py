import pandas as pd
import tensorflow as tf
import os
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from numpy import reshape

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_FILE = os.path.join(BASE_DIR, "receipt_count_prediction_model.h5")
DATA_FILE = os.path.join(BASE_DIR, "data_daily.csv")

# Load the saved model
loaded_model = tf.keras.models.load_model(MODEL_FILE)

# Load daily data
with open(DATA_FILE, "r") as f:
    df_raw = pd.read_csv(f,index_col=0)


def predict(day_str):
    # Get previous day receipt count
    previous_day_data = getPreviousData(day_str)

    # Predict the next day receipt count
    if previous_day_data is not None:
        # normalize and prepare the data as in training
        scaler = MinMaxScaler(feature_range=(0, 1))
        prev_data = previous_day_data.reshape(-1, 1) # to make 2D array required by fit_transform
        prev_data = scaler.fit_transform(prev_data)
        prev_data = reshape(prev_data, (prev_data.shape[0], 1, prev_data.shape[1]))

        # predict
        scaled_data_predicted = loaded_model.predict(prev_data)

        # return invert prediction
        return int(scaler.inverse_transform(scaled_data_predicted)[0][0])
    else:
        return -1
    

def getPreviousData(day_str):
    try:
        # Convert the date string to a datetime object
        day = datetime.strptime(day_str, '%Y-%m-%d')

        # Subtract one day
        previous_day = day - timedelta(days=1)

        # Convert the result back to a string in the same format
        previous_day_str = previous_day.strftime('%Y-%m-%d')

        # Find previous day data if it exists
        if previous_day_str in df_raw.index:
            df_previous_day = df_raw[df_raw.index == previous_day_str]
            # Return the count for previous day
            return df_previous_day['Receipt_Count'].values
        else:
            # Handling the case where data for the previous day is not available
            return None
    except Exception as e:
        # Handle exceptions (e.g., invalid date string, file not found)
        print(f"An error occurred: {e}")
        return None

#if __name__ == "__main__":
    # Define the date string for which you want to retrieve previous data
#    input_date = '2022-01-01'

    # Call the function and print the result
#    result = predict(input_date)
#    print(result)    
#    if result >= 0:
#        print(f"Receipt count prediction for ({input_date}): {result}")
#    else:
#        print("No data available for the specified date.")