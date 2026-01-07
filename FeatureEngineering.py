import pandas as pd
import numpy as np

def main():
    features = [
    'Timestamp',
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
    'Traffic_Condition'
    ]

    #novas colunas de features temporais
    time_features = [
    'Hour',
    'Day_of_Week',
    'Hour_sin',
    'Hour_cos',
    'Day_sin',
    'Day_cos'
    ]

    target = ['Traffic_Speed_kmh']
    all_columns = features + target
    output_file = 'D:/IC_Codes/features_engineered.csv'
    input_file = 'D:/IC_Codes/smart_mobility_dataset.csv'

    df = pd.read_csv(
        input_file,
        usecols=all_columns,
        na_values='?',
        delimiter=','
    )

    df = extract_time_features(df)

    final_columns = [
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
    'Traffic_Condition'
    ] + time_features + target

     # Salva arquivo com o tratamento dos dados e as novas features
    df.to_csv(output_file, columns = final_columns,  header=True, index=False)  

def extract_time_features(df):
    # Converte Timestamp para datetime
    df['Timestamp'] = pd.to_datetime(
    df['Timestamp'],
    format='%Y-%m-%d %H:%M:%S',
    errors='coerce'
    )
    
    # Extrai hora do dia (0-23)
    df['Hour'] = df['Timestamp'].dt.hour

    # Extrai dia da semana (0=segunda, 6=domingo)
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

    # Codificação cíclica da hora
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    # Codificação cíclica do dia da semana
    df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)

    # Remove Timestamp original 
    df.drop(columns=['Timestamp'], inplace=True)

    return df

if __name__ == "__main__":
    main()