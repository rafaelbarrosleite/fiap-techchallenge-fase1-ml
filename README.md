# Tech Challenge - Fase 1 - Diagnóstico de Câncer de Mama com Machine Learning

Esse projeto é a minha entrega do Tech Challenge da Fase 1 da pós. A ideia foi construir a base de um sistema de apoio ao diagnóstico para um hospital: a partir dos dados de um exame, o modelo tenta prever se um tumor de mama é **benigno** ou **maligno**, ajudando na triagem e dando uma informação a mais para o médico (que sempre tem a palavra final).

## O problema

O hospital recebe um volume grande de exames e precisa de uma forma de acelerar a triagem sem perder qualidade. Aqui eu trato isso como um problema de **classificação binária**: dado um conjunto de medidas do exame, classificar o tumor em benigno (0) ou maligno (1).

## Dataset

Usei a base **Breast Cancer Wisconsin (Diagnostic)**, que é pública e bem conhecida.

- Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- São 569 registros e 30 variáveis numéricas (medidas do núcleo das células, como raio, textura, perímetro, área, etc.).
- A coluna alvo é `diagnosis` (M = maligno, B = benigno).
- O arquivo `data.csv` já está incluído neste repositório.

## Estrutura do projeto

```
.
├── Tech_Challenge_B.ipynb     # Notebook principal com toda a análise e os modelos
├── data.csv                   # Base de dados
├── requirements.txt           # Bibliotecas necessárias
├── Dockerfile                 # Para rodar o projeto via Docker
├── relatorio_tecnico.md       # Relatório técnico (pode ser exportado pra PDF)
└── README.md                  # Este arquivo
```

## Modelos usados

Testei três algoritmos de classificação e comparei os resultados:

- **Regressão Logística**
- **Random Forest**
- **KNN (K-Nearest Neighbors)**

A métrica que eu priorizei foi o **Recall da classe maligna**, porque em diagnóstico o erro mais grave é deixar passar um tumor maligno (falso negativo).

## Como executar

### Opção 1 - Rodando localmente (sem Docker)

Pré-requisito: Python 3.10 ou superior instalado.

```bash
# 1. (Opcional, mas recomendado) criar um ambiente virtual
python -m venv venv
source venv/bin/activate      # No Windows: venv\Scripts\activate

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o notebook
jupyter notebook
```

Depois é só abrir o arquivo `Tech_Challenge_B.ipynb` e rodar as células de cima pra baixo.

### Opção 2 - Rodando com Docker

Pré-requisito: Docker instalado.

```bash
# 1. Construir a imagem
docker build -t tech-challenge-fase1 .

# 2. Subir o container
docker run -p 8888:8888 tech-challenge-fase1
```

Vai aparecer no terminal um link com um token (algo como `http://127.0.0.1:8888/?token=...`). É só abrir esse link no navegador e abrir o notebook.

## Resultados (resumo)

Os modelos chegaram a uma acurácia em torno de 95% a 98% nesse dataset. O modelo escolhido como final foi a **Regressão Logística**, por dois motivos: teve um ótimo recall na classe maligna e é mais fácil de interpretar, o que é importante num contexto médico. Os detalhes, gráficos e a discussão completa estão no notebook e no `relatorio_tecnico.md`.

## Observação importante

Esse modelo é uma ferramenta de **apoio**. Ele não substitui o diagnóstico médico em nenhuma hipótese. A decisão final é sempre do profissional de saúde.
