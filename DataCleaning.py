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

    #Separa colunas categóricas e numéricas
    numeric_features = df[features].select_dtypes(include=[np.number]).columns
    categorical_features = df[features].select_dtypes(exclude=[np.number]).columns

    print("Features Numéricas:\n")
    print (numeric_features)
    print("\n")
    print("Features Categóricas:\n")
    print(categorical_features)
    print("\n")

    # Numéricos: Mediana (robusto contra outliers)
    for col in numeric_features:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Categóricos: Moda (mais frequente)
    for col in categorical_features:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

    #Imprime a quantidade de valores faltantes após a limpeza
    print("VALORES FALTANTES APÓS A LIMPEZA\n")   
    print(df.isnull().sum())
    
    # Salva arquivo com o tratamento para dados faltantes
    df.to_csv(output_file, header=True, index=False)  
    

if __name__ == "__main__":
    main()