import pandas as pd

def main():
    features = [
    'Latitude',
    'Longitude',
    'Vehicle_Count',
    'Road_Occupancy_%',
    'Traffic_Light_State',
    'Weather_Condition',
    'Accident_Report',
    'Sentiment_Score',
    'Ride_Sharing_Demand',
    'Parking_Availability',
    'Emission_Levels_g_km',
    'Energy_Consumption_L_h',
    'Traffic_Condition',
    'Hour',
    'Day_of_Week',
    'Hour_sin',
    'Hour_cos',
    'Day_sin',
    'Day_cos'
    ]

    target = ['Traffic_Speed_kmh']
    all_columns = features + target

    output_file = 'D:/IC_Codes/one-hot_encoded.csv'
    input_file = 'D:/IC_Codes/features_engineered.csv'

    df = pd.read_csv(
        input_file,
        usecols=all_columns,
        delimiter=','
    )

    df = one_hot_encoding(df)
    print(df)

    df.to_csv(output_file, header=True, index=False)  

def one_hot_encoding(df):
    df = pd.get_dummies(
        df,
        columns=['Weather_Condition', 'Traffic_Light_State'],
        drop_first=False, 
        dtype=int
    )
    return df

if __name__ == "__main__":
    main()