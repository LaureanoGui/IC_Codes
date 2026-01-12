from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

def main():
    features = [
    'Latitude',
    'Longitude',
    'Vehicle_Count',
    'Road_Occupancy_%',
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
    'Day_cos',
    'Weather_Condition_Clear',
    'Weather_Condition_Fog',
    'Weather_Condition_Rain',
    'Weather_Condition_Snow',
    'Traffic_Light_State_Green',
    'Traffic_Light_State_Red',
    'Traffic_Light_State_Yellow'
    ]

    target = ['Traffic_Speed_kmh']
    all_columns = features + target

    output_file = 'D:/IC_Codes/ordinal_encoded.csv'
    input_file = 'D:/IC_Codes/one-hot_encoded.csv'

    df = pd.read_csv(
        input_file,
        usecols=all_columns,
        delimiter=','
    )

    #Mostra todos os valores distintos que aparecem na coluna antes do ordinal-encoding
    print("VALORES DISTINTOS: \n")
    print(df['Traffic_Condition'].unique())
    print("\n")

    #Conta quantas vezes cada categoria aparece na coluna antes do ordinal-encoding
    print("QUANTIDADE DE VEZES EM QUE APARECEM: \n")
    print(df['Traffic_Condition'].value_counts())
    print("\n")
    df = ordinal_encoding(df)

     #Mostra todos os valores distintos que aparecem na coluna Após o ordinal-encoding
    print("VALORES DISTINTOS (ORDINAL-ENCODING): \n")
    print(df['Traffic_Condition'].unique())
    print("\n")

    df.to_csv(output_file, header=True, index=False)  

encoder = OrdinalEncoder(
    categories=[['Low', 'Medium', 'High']]
)

def ordinal_encoding(df):
    df[['Traffic_Condition']] = encoder.fit_transform(
    df[['Traffic_Condition']]
    )
    return df

if __name__ == "__main__":
    main()



