import pandas as pd
import numpy as np
def main():
    # Faz a leitura do arquivo

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

    target = ['Traffic_Speed_kmh']

    all_colums = features + target

    output_file = 'D:/IC_Codes/cleanData.csv'
    input_file = 'D:/IC_Codes/smart_mobility_dataset.csv'
    df = pd.read_csv(input_file,         # Nome do arquivo com dados
                     usecols = all_colums, # Define as colunas que serão  utilizadas
                     na_values='?', delimiter=',')      # Define que ? será considerado valores ausentes      
    
    df_original = df.copy()
    # Imprime as 15 primeiras linhas do arquivo
    print("PRIMEIRAS 15 LINHAS\n")
    print(df.head(15))
    print("\n")        

    # Imprime informações sobre dos dados
    print("INFORMAÇÕES GERAIS DOS DADOS\n")
    print(df.info())
    print("\n")
    
    # Imprime uma analise descritiva sobre dos dados
    print("DESCRIÇÃO DOS DADOS\n")
    print(df.describe())
    print("\n")
    
    # Imprime a quantidade de valores faltantes por coluna
    print("VALORES FALTANTES\n")
    print(df.isnull().sum())
    print("\n")  

    # Verificando valores faltantes no target
    print("VALORES FALTANTES NO TARGET\n")
    print(df[target].isnull().sum())
    print("\n")

    # Remove linhas onde o target é NaN
    df = df.dropna(subset=target)

    
    columns_missing_value = df[features].columns[df[features].isnull().any()]
    print(columns_missing_value)
    method = 'mode' # number or median or mean or mode
    
    for c in columns_missing_value:
        UpdateMissingValues(df, c)
    
    print(df.describe())
    print("\n")
    print(df.head(15))
    print(df_original.head(15))
    print("\n")
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=True, index=False)  
    
def UpdateMissingValues(df, column, method="mode", number=0):
    if method == 'number':
        # Substituindo valores ausentes por um número
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        # Substituindo valores ausentes pela mediana 
        median = df[column].median()
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        # Substituindo valores ausentes pela média
        mean = df[column].mean()
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        # Substituindo valores ausentes pela moda
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)


if __name__ == "__main__":
    main()