import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pearson_heatmap():
    input_file = 'D:/IC_Codes/ordinal_encoded.csv'

    # Leitura do dataset
    df = pd.read_csv(input_file)

    # Calcula a matriz de correlação de Pearson
    correlation_matrix = df.corr(method='pearson')

    # Configuração do tamanho da figura
    plt.figure(figsize=(14, 10))

    # Criação do heatmap
    sns.heatmap(
        correlation_matrix,
        annot=True,       
        cmap='coolwarm',
        linewidths=0.5
    )

    plt.title('Heatmap da Correlação de Pearson')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pearson_heatmap()
