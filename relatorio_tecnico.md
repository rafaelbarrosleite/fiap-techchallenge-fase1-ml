# Relatório Técnico - Tech Challenge Fase 1
## Diagnóstico de Câncer de Mama com Machine Learning

**Autor:** Rafael de Barros Leite — rafaelbarrosleite@gmail.com | rafael.leite@gruposbf.com.br

---

## 1. O problema

A proposta do desafio foi imaginar um hospital universitário que recebe muitos exames e precisa de um sistema de IA pra ajudar na triagem. Dentro disso, eu escolhi resolver um problema de **classificação**: a partir das medidas de um exame, prever se um tumor de mama é **benigno** ou **maligno**.

A motivação é clara: quanto antes a equipe identificar um caso suspeito de malignidade, mais rápido o paciente pode ser encaminhado. Mas, desde já, deixo registrado que a ideia é ser uma **ferramenta de apoio** — o médico continua tendo a palavra final.

## 2. Dataset escolhido

Usei a base **Breast Cancer Wisconsin (Diagnostic)**, que é pública e bem usada em estudos de ML.

- 569 registros
- 30 variáveis numéricas (medidas como raio, textura, perímetro, área, suavidade, concavidade, etc., calculadas a partir de imagens das células)
- Coluna alvo: `diagnosis` (M = maligno, B = benigno)

Escolhi essa base porque ela é direta, é uma das sugeridas no enunciado e representa bem um problema real de diagnóstico.

## 3. Exploração dos dados (EDA)

Antes de modelar, fui entender com o que eu estava lidando:

- Olhei o formato da base (`shape`), as primeiras linhas (`head`) e os tipos das colunas (`info`).
- Vi as estatísticas descritivas com `describe`.
- Conferi o balanceamento das classes: deu **62,7% benignos e 37,3% malignos**. Ou seja, tem um desbalanceamento, mas não é muito forte.
- Plotei histogramas (radius_mean, texture_mean, area_mean, perimeter_mean) e boxplots por diagnóstico.

O que eu percebi: as variáveis ligadas ao **tamanho e à forma do tumor** (raio, perímetro, área e concavidade) tendem a ter valores bem mais altos nos casos malignos. Isso já dá um indício de que essas variáveis vão ser importantes pro modelo.

## 4. Estratégias de pré-processamento

Esse dataset é bem limpo, então não precisei fazer muita coisa pesada. O que eu fiz:

- **Remoção de colunas inúteis:** removi a coluna `Unnamed: 32`, que estava completamente vazia (era só um artefato do CSV), e a coluna `id`, que é só um identificador e não ajuda na previsão.
- **Valores ausentes:** conferi com o `missingno` e com `isnull().sum()`. Tirando a coluna vazia que removi, **não tinha nenhum valor ausente** nas variáveis de fato. Então não precisei fazer imputação.
- **Variável categórica:** a única categórica era o alvo `diagnosis`. Converti com `map`: B = 0 e M = 1. Coloquei o maligno como 1 de propósito, porque ele é a classe "positiva" que eu mais quero detectar.
- **Outliers:** apareceram alguns nos boxplots, mas decidi **manter**, porque valores extremos aqui geralmente são justamente os casos malignos. Se eu removesse, ia jogar fora informação importante.
- **Escala (normalização):** apliquei `StandardScaler` nos modelos que precisam (Regressão Logística e KNN). No Random Forest não precisa, porque árvore não se importa com escala. Importante: coloquei o scaler **dentro do pipeline**, então ele aprende a média e o desvio só com os dados de treino, evitando vazamento de dados (data leakage).
- **Correlação:** fiz a matriz de correlação e também a correlação de cada variável com o alvo. Confirmou o que vi na EDA: raio, perímetro, área e concavidade são as mais correlacionadas com o diagnóstico.

## 5. Separação treino / validação / teste

Separei os dados em três partes, usando `stratify` pra manter a proporção das classes em cada pedaço:

- **Treino:** 60% (341 registros)
- **Validação:** 20% (114 registros)
- **Teste:** 20% (114 registros)

Na prática, o 60/20/20 sai de dois cortes: primeiro separo 20% pro teste (sobram 80%), e depois pego 25% desses 80% pra validação (25% de 80% = 20% do total), sobrando 60% pro treino.

Usei a validação pra comparar os modelos entre si e escolher o melhor, e deixei o **teste só pro final**, pra medir o desempenho do modelo escolhido em dados que ele nunca viu.

## 6. Modelos usados e por quê

Testei três algoritmos diferentes:

- **Regressão Logística** — modelo simples, rápido e bem interpretável. Pra um contexto médico, conseguir explicar o porquê de uma previsão é uma vantagem grande.
- **Random Forest** — modelo mais robusto, lida bem com relações não lineares e me dá a importância das variáveis de graça.
- **KNN** — um modelo bem diferente dos outros dois, baseado em distância, pra ter um terceiro ponto de comparação.

## 7. Escolha da métrica

Esse ponto eu acho o mais importante do trabalho. Em diagnóstico de câncer, **nem todo erro é igual**:

- Um **falso positivo** (dizer que é maligno quando é benigno) gera um susto e exames a mais, mas é contornável.
- Um **falso negativo** (dizer que é benigno quando na verdade é maligno) é muito mais grave, porque pode atrasar o tratamento de um paciente que precisava.

Por isso, a métrica que eu priorizei foi o **Recall da classe maligna** — ele mede quantos dos casos malignos reais o modelo conseguiu pegar. Eu olho a acurácia e o F1 também, mas o recall é o que manda na decisão. Acurácia sozinha pode enganar num dataset desbalanceado.

## 8. Treinamento, avaliação e resultados

Treinei cada modelo no conjunto de treino e avaliei na validação. Resumo dos resultados na validação:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Regressão Logística | 0,9737 | 0,9545 | **0,9767** | 0,9655 |
| Random Forest | 0,9737 | 0,9762 | 0,9535 | 0,9647 |
| KNN | 0,9737 | 1,0000 | 0,9302 | 0,9639 |

Os três modelos ficaram empatados em acurácia (0,9737), mas, olhando pelo **Recall da classe maligna** (a métrica que eu priorizei), a **Regressão Logística saiu na frente** (0,9767 contra 0,9535 do Random Forest e 0,9302 do KNN). Por isso escolhi ela como modelo final — e ainda por cima ela é a mais fácil de interpretar, o que pesa num contexto de saúde.

No **conjunto de teste** (dados nunca vistos), o modelo final teve:

- Accuracy: **0,9737** (~97,4%)
- Recall (maligno): **0,9524** (~95,2%)
- F1-score: **0,9639** (~96,4%)

A matriz de confusão do teste mostra que, dos 42 tumores malignos, o modelo **acertou 40 e deixou passar 2** (2 falsos negativos), além de 1 falso positivo. Esse número de falsos negativos é justamente o que eu mais quero acompanhar, porque é o erro mais perigoso no problema.

## 9. Interpretação dos resultados (Feature Importance e SHAP)

Pra entender o que o modelo está "olhando", usei duas técnicas:

- **Feature Importance** (no Random Forest): as variáveis mais importantes foram as ligadas a tamanho e forma do tumor (`concave points`, `perimeter`, `radius`, `area` nas versões `_worst` e `_mean`).
- **SHAP:** confirmou a mesma história e ainda mostra a direção — valores altos dessas variáveis empurram a previsão pra "maligno". Também consegui ver, com o gráfico de waterfall, a explicação de um paciente individual, o que é bem útil pra mostrar pro médico o porquê de cada previsão.
- **Coeficientes da Regressão Logística** (modelo final): bateram com a mesma lógica — as variáveis de tamanho e forma do tumor são as que mais empurram a previsão pra maligno.

Isso bate com o conhecimento clínico (tumores malignos tendem a ser maiores e com bordas mais irregulares), o que me deixa mais confiante de que o modelo não está aprendendo besteira.

## 10. Discussão crítica - dá pra usar na prática?

Na minha visão, **dá pra usar, mas como apoio, nunca como decisão final**. Alguns pontos:

- **Como usaria:** o modelo poderia rodar junto com o exame e sinalizar os casos com alta probabilidade de malignidade pra serem priorizados na fila do médico. Funcionaria como um "segundo olhar" automático.
- **Cuidados:** o dataset é relativamente pequeno e de uma fonte específica. Antes de usar de verdade, precisaria validar com dados do próprio hospital e acompanhar se o desempenho se mantém. E aqueles 2 falsos negativos no teste mostram que o modelo não é perfeito — ele erra, e em diagnóstico isso é sério.
- **Responsabilidade:** o modelo erra, e em saúde erro tem consequência. Por isso o médico precisa sempre revisar. O sistema ajuda a ganhar tempo e a não deixar passar caso suspeito, mas a palavra final é sempre humana.

## 11. Conclusão

Consegui montar a base do sistema pedido: carreguei e explorei os dados, fiz o pré-processamento, treinei e comparei três modelos, escolhi o melhor com base numa métrica que faz sentido pro problema (recall) e interpretei os resultados. O modelo final (Regressão Logística) teve uma acurácia de ~97% e recall de ~95% na classe maligna e, com os cuidados certos, pode servir como ferramenta de triagem e apoio ao diagnóstico.
