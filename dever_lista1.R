# LISTA 1
# Guilherme Braga

# Exercícios da matéria de Redes Neurais 1 do Professor Guilherme Rodrigues
# Mestrado em Estatística - Universidade de Brasília
# Matéria de Semestre 2024/01
# Feito no final de 2025 para estudo pessoal

# https://github.com/janishar/mit-deep-learning-book-pdf

# Dependências
library(latex2exp)
library(NeuralNetTools)
library(tidyverse)
library(microbenchmark)

# "Nesta lista estamos interessados em estimar o modelo acima usando uma rede neural simples, ajustada
# sobre os dados simulados. Precisamente, queremos construir uma rede neural com apenas uma camada
# escondida contendo dois neurônios."

### Arquitetura da RNA (como dado na lista de exercícios)
par(mar=c(0, 0, 0, 0))
wts_in <- rep(1, 9)
struct <- c(2, 2, 1) # dois inputs, dois neurônios escondidos e um otput
plotnet(wts_in, struct = struct,
        x_names="", y_names="",
        node_labs=F, rel_rsc=.7)
aux <- list(
  x=c(-.8, -.8, 0, 0, .8, rep(-.55, 4), -.12, -0.06, .38, .38, .7),
  y=c(.73, .28, .73, .28, .5, .78, .68, .48, .32, .88, .5, .68, .44, .7),
  rotulo=c("x_1", "x_2", "h_1", "h_2", "\\hat{y}", paste0("w_", 1:4),
           "b_1", "b_2", "w_5", "w_6", "b_3")
)
walk(transpose(aux), ~ text(.$x, .$y,
                            TeX(str_c("$", .$rotulo, "$")), cex=.8))

# -----------------------------------------------------

# Como dado na questão

set.seed(1.2024)

m.obs <- 100000

dados <- tibble(
  x1 = runif(m.obs, -3, 3),
  x2 = runif(m.obs, -3, 3)
) %>%
  mutate(
    mu = abs(x1^3 - 30*sin(x2) + 10),
    y  = rnorm(m.obs, mean = mu, sd = 1)
  )

# -----------------------------------------------------

# ITEM A

# Como no algoritmo 6.3 (Livro Deep Learning, MIT, Ian Goodfellow)
# https://mcube.lab.nycu.edu.tw/~cfung/docs/books/goodfellow2016deep_learning.pdf
# Página 227

# calcular y_chapeu = funcão (x_ji e theta)
# de j features e i observacoes

# f(x e theta) = sigmoide(x_1i*w1 + x_2i*w3 + b1) * w5 
          #   + sigmoide(x_1i*w2 + x_2i*w4 + b2) * w6
          # + b3


# como declarada na funcao de ativacão
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Requisitos, iniciar parâmetros da rede neural e saber o tamanho da rede

# w1, w2, ..., w9
pesos_iniciais <- rep(0.1, 6)

# b1, b2, b3
bias_iniciais <- rep(0.1, 3)

# vetor de input x: 1, 1
x <- c(1, 1)

# funcão de predicão da variavel resposta
predict_nn <- function(x, pesos, bias) {
  
  # desempacotar pesos
  w1 <- pesos[1]; w2 <- pesos[2]
  w3 <- pesos[3]; w4 <- pesos[4]
  w5 <- pesos[5]; w6 <- pesos[6]
  
  # desempacotar bias
  b1 <- bias[1]; b2 <- bias[2]; b3 <- bias[3]
  
  # funcão y_chapeu
  a1 <- x[1]*w1 + x[2]*w3 + b1
  a2 <- x[1]*w2 + x[2]*w4 + b2
  
  h1 <- sigmoid(a1)
  h2 <- sigmoid(a2)
  
  y_chapeu <- h1*w5 + h2*w6 + b3
  
  return(list(
    y_chapeu = y_chapeu,
    h1    = h1,
    h2    = h2, 
    a1    = a1, 
    a2    = a2
  ))
  
}

# O resultado que sai aqui seria o primeiro "forward pass" da rede
result_forward <- predict_nn(x, pesos = pesos_iniciais, bias = bias_iniciais)
y_chapeu <- result_forward$y_chapeu

# -----------------------------------------------------

# ITEM B

# "Funcão de Custo", Erro Quadratico Médio

# "primeiras 80.000 amostras componham o conjunto de treinamento"
dados_train <- dados[1:80000, ]

# próximas 10.000 o de validação
dados_val   <- dados[80001:90000, ]

# últimas 10.000 o de teste
dados_test  <- dados[90001:100000, ]

# Qual é o custo da rede no conjunto de teste quando θ = (0.1, . . . , 0.1)?
# Sendo θ (theta) = (w1, . . . , w6, b1, b2, b3), ou seja, pesos e bias

custo_nn <- function(pesos_custo, bias_custo, X, y) {
  res <- apply(X, 1, predict_nn, pesos = pesos_custo, bias = bias_custo)
  y_chapeu <- sapply(res, `[[`, "y_chapeu") 
  mean((y - y_chapeu)^2)
}


# apply(obj, MARGIN, FUN, ...)
# margin = 1 (por linha) ou 2 (por coluna)
# obj é a matriz/array que vai aplicar
# o resto são parametros da funcão


# No nosso caso, y é o amostrado daquela funcão que queremos aproximar
y_test <- dados_test$y

# E x é os dados de teste (x1 e x2) só que transformados para matriz

x_test <- as.matrix(dados_test[, c("x1", "x2")])

# Os parâmetros ainda são os iniciais

# w1, w2, ..., w9
pesos_iniciais <- rep(0.1, 6)

# b1, b2, b3
bias_iniciais <- rep(0.1, 3)

custo_teste <- custo_nn(
  pesos_custo = pesos_iniciais,
  bias_custo  = bias_iniciais,
  X = x_test,
  y = y_test
)

# -----------------------------------------------------

# ITEM C

# Use a regra da cadeia para encontrar expressões algébricas para o vetor gradiente

# Meus rabiscos desse resultado estão anexados em PDF, mas é um resultado relativamente simples de achar na internet e no livro do Goodfellow. 
# https://medium.com/@ppuneeth73/the-chain-rule-of-calculus-the-backbone-of-deep-learning-backpropagation-9d35affc05e7
# Essa seria a explicacão mais "alto nível" da coisa.

# -----------------------------------------------------

# ITEM D

# Back Propagation
# https://mcube.lab.nycu.edu.tw/~cfung/docs/books/goodfellow2016deep_learning.pdf
# Página 228

# Vou implementar as continhas que eu fiz ali atrás no item C. 

y_treino <- dados_train$y

# Isso aqui é como se fosse apenas 1 iteracão do backpropagation!
# to pegando os primeiros valores de x1 e x2 da rede neural, na primeira iteracão
x <- c(dados_train$x1[1], dados_train$x2[1])

# calculando o resultado da rede, y_chapeu
# e aproveitando para pegar h1 e h2
out <- predict_nn(x, pesos_iniciais, bias_iniciais)
y_chapeu <- out$y_chapeu
h1 <- out$h1
h2 <- out$h2

# peso 6
dJ_dw6 <- ( 2 * (y_chapeu - y_treino[1]) * h1 )

# peso 5
dJ_dw5 <- ( 2 * (y_chapeu - y_treino[1]) * h2 )

# bias 3
dJ_db3 <- ( 2 * (y_chapeu - y_treino[1]))


# peso 1
dJ_dw1 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[5] * h1 * (1-h1) * x[1]

# peso 3
dJ_dw3 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[5] * h1 * (1-h1) * x[2]

# bias 1
dJ_db1 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[5] * h1 * (1-h1)


# peso 2
dJ_dw2 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[6] * h2 * (1-h2) * x[1]

# peso 4
dJ_dw4 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[6] * h2 * (1-h2) * x[2]

# bias 2
dJ_db2 <- ( 2 * (y_chapeu - y_treino[1])) * pesos_iniciais[6] * h2 * (1-h2) 


# legal? legal. Fez sentido. 
# Vou fazer pra todo mundo e não só para 1 item
# No final é a mesma coisa de antes mas eu to tirando a média com o "mean" 
# grad_final = 1/m dos "m" observacoes de todas essas derivadas 

# w1, w2, ..., w9
pesos_iniciais <- rep(0.1, 6)

# b1, b2, b3
bias_iniciais <- rep(0.1, 3)

# dados de treino
X_train <- as.matrix(dados_train[, c("x1", "x2")])

# Outputs de treino para aplicar no erro quadratico medio
y_treino <- dados_train$y

grad_nn <- function(pesos, bias, X, y) {
  
  out <- predict_nn(x, pesos, bias)
  y_chapeu <- out$y_chapeu
  
  h1 <- out$h1 # valor na rede - h1
  h2 <- out$h2 # valor na rede - h2
  
  # as partes de y_chapeu
  a1 <- out$a1 # x[1]*w1 + x[2]*w3 + b1
  a2 <- out$a2 # x[1]*w2 + x[2]*w4 + b2
  
  x1 <- X[, 1]
  x2 <- X[, 2]
  
  # BACKWARD (uma vez só)
  delta3 <- 2 * (y_chapeu - y)                 
  
  dw5 <- mean(delta3 * h1)
  dw6 <- mean(delta3 * h2)
  db3 <- mean(delta3)
  
  delta1 <- delta3 * pesos[5] * h1 * (1 - h1)    
  delta2 <- delta3 * pesos[6] * h2 * (1 - h2)     
  
  dw1 <- mean(delta1 * x1)
  dw3 <- mean(delta1 * x2)
  db1 <- mean(delta1)
  
  dw2 <- mean(delta2 * x1)
  dw4 <- mean(delta2 * x2)
  db2 <- mean(delta2)
  
  list(
    grad_pesos = c(dw1, dw2, dw3, dw4, dw5, dw6),
    grad_bias  = c(db1, db2, db3),
    cache = list(y_chapeu = y_chapeu, h1 = h1, h2 = h2, a1 = a1, a2 = a2)
  )
}

g <- grad_nn(pesos_iniciais, bias_iniciais, X_train, y_treino)

g$grad_pesos
g$grad_bias

# -----------------------------------------------------

# ITEM E

# agora sim, aplicar o método de gradiente com
# taxa de aprendizagem = 0.1
# 100 iteraões minimizando a funcão de custo

# a atualizacão dos pesos/bias é dado por
# pesos = pesos - * (taxa_aprendizagem * gradiente_pesos)
# bias = bias - * (taxa_aprendizagem * gradiente_bias)

X_treino <- as.matrix(dados_train[, c("x1","x2")])
y_treino <- dados_train$y

X_val <- as.matrix(dados_val[, c("x1","x2")])
y_val <- dados_val$y

eps <- 0.1
n_iter <- 100

pesos <- rep(0, 6)
bias  <- rep(0, 3)

hist_train <- numeric(n_iter)
hist_val   <- numeric(n_iter)

best_val <- Inf
best_iter <- NA
best_pesos <- NULL
best_bias  <- NULL

for (t in 1:n_iter) {
  # gradiente no treino
  g <- grad_nn(pesos, bias, X_treino, y_treino)
  
  # atualização (descida)
  pesos <- pesos - eps * g$grad_pesos
  bias  <- bias  - eps * g$grad_bias
  
  # custos (para monitorar)
  hist_train[t] <- custo_nn(pesos, bias, X_treino, y_treino)
  hist_val[t]   <- custo_nn(pesos, bias, X_val, y_val)
  
  # melhor validação
  if (hist_val[t] < best_val) {
    best_val <- hist_val[t]
    best_iter <- t
    best_pesos <- pesos
    best_bias  <- bias
  }
  
  #print(t)
  
}

# dai aqui é os resultados da questão
best_val # melhor resultado de custo foi 1.79
best_iter # melhor iteracão foi a 3 (eita ta certo isso)
best_pesos # 1.045710  1.045710 -3.599251 -3.599251  4.568202  4.568202
best_bias # 3.698907 3.698907 9.809250


# -----------------------------------------------------

# ITEM F

# Plotar esse negócio com grafico

# Não entendo muito de R, o chatgpt me salvou nesse tipo de implementacão
# Na verdade, fica essa a explicacão caso o código esteja feio, ainda aprendendo enquanto programo
# E o chatgpt me ajuda a debugar :)

plot(
  hist_train,
  type = "l",
  lwd = 2,
  col = "blue",
  xlab = "Iteração",
  ylab = "Custo (MSE)",
  ylim = range(c(hist_train, hist_val)),
  main = "Custo no treino e validação"
)

lines(
  hist_val,
  lwd = 2,
  col = "red",
  lty = 2
)

legend(
  "topright",
  legend = c("Treino", "Validação"),
  col = c("blue", "red"),
  lwd = 2,
  lty = c(1, 2),
  bty = "n"
)

# interpretacão do queridissimo ChatGPT
# Curva azul — Treino
# Cai muito rápido.
# Converge para um custo praticamente zero.
# Indica que a rede memoriza o conjunto de treino com facilidade.

# Curva vermelha — Validação
# Primeiro cai,
# depois sobe rapidamente,
# e satura em um patamar alto.
#Isso não é ruído. É comportamento sistemático.

# Resultado - Parece ser um overfitting severo
# Jtrain ABAIXA enquanto Jval AUMENTDA

# Isso é esperado:

# rede pequena,
# muitos dados,
# função alvo determinística.
# Seria esse o caso de um early stopping! Para generalizar e não decorar os dados

# -----------------------------------------------------

# ITEM G

# -----------------------------------------------------

# ITEM H

# -----------------------------------------------------

# ITEM I

# -----------------------------------------------------

# ITEM J

# -----------------------------------------------------

# ITEM K

# -----------------------------------------------------

# ITEM L

# -----------------------------------------------------

# ITEM M

# -----------------------------------------------------








