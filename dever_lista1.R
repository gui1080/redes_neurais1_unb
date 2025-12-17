# Lista 1
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
  
  # funćão y_chapeu
  a1 <- x[1]*w1 + x[2]*w3 + b1
  a2 <- x[1]*w2 + x[2]*w4 + b2
  
  h1 <- sigmoid(a1)
  h2 <- sigmoid(a2)
  
  y_chapeu <- h1*w5 + h2*w6 + b3
  y_chapeu
}

# O resultado que sai aqui seria o primeiro "forward pass" da rede
y_chapeu <- predict_nn(x, pesos = pesos_iniciais, bias = bias_iniciais)

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
  y_chapeu <- apply(X, 1, predict_nn, pesos = pesos_custo, bias = bias_custo)
  mean((y - y_chapeu)^2) # erro quadratico médio do real (y) e o observado (y_chapeu)
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

custo_teste <- custo_nn(pesos_custo = pesos_iniciais, bias_custo = bias_iniciais, x_test, y_test)

# -----------------------------------------------------

# ITEM C

# Use a regra da cadeia para encontrar expressões algébricas para o vetor gradiente

# -----------------------------------------------------

# ITEM D

# Back Propagation
# https://mcube.lab.nycu.edu.tw/~cfung/docs/books/goodfellow2016deep_learning.pdf
# Página 228

# -----------------------------------------------------

# ITEM E

# -----------------------------------------------------

# ITEM F

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








