# 🛒 Análise de Propensão de Compra em E-Commerce (Churn/Revenue)

## 1. A Primeira Análise

Foquei inicialmente em analisar a variável-alvo, que é a coluna chamada `'Revenue'`. Com a biblioteca `matplotlib` e `seaborn`, tratei de entender quantas compras (em percentual) de fato se tornaram negócios fechados, isto é, quando o cliente tem o produto em seu carrinho, qual a porcentagem que certamente fechou o negócio.

Inicialmente, com essa primeira EDA, temos o primeiro resultado:

> **Distribuição de Compras E-Commerce (Em %):**
> * **False (Não gerou receita):** 84.5%
> * **True (Gerou receita):** 15.5%

Observa-se o quão os dados estão desbalanceados, sendo apenas 15,5% dos negócios fechados (True), contra 84,5% que não geraram receita, ou seja, False. Isso já é um indício que os futuros modelos de Machine Learning deverão ser extremamente bem treinados para conseguir encontrar esse percentual que é bem pequeno. Assim, o cuidado com overfitting é um dos pontos de atenção até aqui.

---

## 2. O Problema de Negócio na Prática: O que faz alguém comprar um produto?

Com nossos dados e através de pesquisas mais aprofundadas sobre o negócio em si, podemos analisar o **Page Value** de uma página de um site. Isso significa que através desta métrica podemos descobrir o valor financeiro médio que uma página brought para o site em si antes de o usuário fechar a compra. A ideia é também entender se este tipo de página realmente consegue atrair e levar o cliente a, de fato, comprar os produtos. 

Para isso, vamos fazer uso de um box-plot para melhorar nossa interpretação. Vejamos a seguir (lembrando que 'True' é quando a pessoa de fato fecha negócio e compra um produto no site analisado, enquanto que 'False' é o contrário):

* **A Caixinha do False:** Repare que no lado 'False' (quem não comprou), a caixinha está completamente esmagada lá no zero. Isso significa que a imensa maioria esmagadora das sessões que não geram receita passam por páginas que têm valor zero para o Google Analytics.
* **A Caixinha do True:** No lado 'True' (quem comprou), a caixinha verde está ligeiramente mais alta, com uma distribuição a mais. Ou seja, pessoas que compram costumam navegar por páginas que têm algum valor atribuído.
* **As Bolinhas (Outliers):** Note que em ambos os lados existem várias bolinhas subindo (chegando até perto de 1000). Isso significa que existem sessões onde o usuário visitou páginas com valores altíssimos. No lado do False, são pessoas que navegaram por páginas muito valiosas (talvez tenham colocado produtos caros no carrinho), mas desistiram da compra.

Assim, essa pode ser uma das colunas 'features' para nosso futuro modelo.

---

### 🗓️ Sazonalidade

Outro ponto que podemos analisar é a sazonalidade, ou seja, será que existem meses onde as pessoas compram mais? Recorrendo ao gráfico de barras, ainda dentro de nossa E.D.A, notamos que:

* **Maio:** Mês do Dia das Mães, as pessoas pesquisam mais, mas não agem com tanto impulso quanto em Novembro.
* **Novembro:** Mês da Black Friday. Veja que o pico de compras fechadas foi muito superior ao mês de maio. Isso sugere que campanhas de retenção pré-novembro podem ser altamente lucrativas.
* **Dezembro:** Parecido com o mês de Maio em termos comportamentais.

---

## 💡 Primeiras Conclusões

Aqui, encerramos a fase 1 do projeto, portanto, já temos algumas colunas que podem ser exploradas com maior afinco e peso dentro de nosso algoritmo de Machine Learning[cite: 1]. A atenção especial até aqui fica para a coluna 'Page Values' e sua relação com a coluna 'Revenue (receita)' — informação que iremos realmente comprovar por meio das métricas de avaliação de nosso modelo[cite: 1]. Outra observação é o real desbalanceamento neste título de problema de negócio, tendo apenas 15,5% do dataset com confirmação de compra[cite: 1]. Isso é característica dos problemas de propensão[cite: 1].

---

## ⚡ O Modelo Escolhido — XGBoost

Até aqui já sabemos que os estudos de Propensão são classificados como problemas de Classificação dentro dos algoritmos de Machine Learning[cite: 1]. Dependendo do tamanho do dataset e de suas particularidades, convém-se escolher modelos mais robustos e refinados, para encontrarmos um bom precision e recall nos dados de teste[cite: 1]. 

O K-Neighbors é um modelo básico de classificação mas, até mesmo por se tratar de um projeto que busca usar ferramentas semelhantes àquelas usadas no dia a dia de um Cientista de Dados, é que fez o **XGBoost** ser escolhido[cite: 1]. Esse é um modelo muito mais refinado e usa as 'Decision Trees — Árvores de Decisão' para melhorar os resultados e métricas do modelo[cite: 1]. A matemática é mais refinada do que a matemática do K-Neighbors e usa conceitos como Derivadas de n ordens para diminuir o erro e melhorar a assertividade do modelo[cite: 1].

---

## 🤖 Modelagem Preditiva & Dados

O objetivo deste resumo não é explorar e detalhar toda a parte de programação, mas sim detalhar o plano de investigação e uso do algoritmo XGBoost[cite: 1].

### Decisão da proporção 80-20 no modelo
Por se tratar de um conjunto de dados com pouco mais de 12 mil linhas, usa-se o padrão de 80% dos dados para treino e 20% separados para teste, usando o conceito de **Stratify** (estratificação), para que a proporção na divisão dos dados se mantenha, ou seja, dentro dos 80% das linhas de treino se mantenham a mesma proporção de 'True' e 'False' que nos dados de Teste (os demais 20%)[cite: 1].

```python
# Iniciando a separacao em um dataframe e uma série (X e y), separando as features da coluna Target.
X = df2.drop(columns=['Revenue'])
y = df2['Revenue'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Quantidade de Linhas para TREINO, o X_train: {X_train.shape[0]}')
print(f'Quantidade de Linhas para o TESTE, o X_test: {X_test.shape[0]}')
