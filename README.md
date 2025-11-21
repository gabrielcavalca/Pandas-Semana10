# Pandas-Semana10

## Comparação de Modelos CNN no Dataset MNIST

Este projeto realiza a comparação entre duas arquiteturas de redes neurais convolucionais (CNN) aplicadas ao dataset MNIST, com o objetivo de avaliar como a profundidade da rede impacta o desempenho na classificação de dígitos manuscritos.

## Resultados

CNN 1 Layer

- Loss: 0.0820
- Accuracy: 0.9796
- Precision: 0.9794410816850345
- Recall: 0.9796523108602152

CNN 3 Layers

- Loss: 0.0445
- Accuracy: 0.9878
- Precision: 0.9877669518857715
- Recall: 0.9878235511891722

Os resultados mostram que o modelo com três camadas convolucionais obtém desempenho superior em todas as métricas avaliadas.

## Descrição do Projeto

O código treina dois modelos CNN utilizando o dataset MNIST obtido via fetch_openml.
Cada modelo é avaliado em termos de loss, accuracy, precision e recall.
As arquiteturas utilizadas são definidas em model.py.

## Como Executar

1. Clone o repositório:

```bash 
git clone https://github.com/SEU_USUARIO/SEU_REPO.git
cd SEU_REPO
```
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o script principal:

```bash
python project.py
```
