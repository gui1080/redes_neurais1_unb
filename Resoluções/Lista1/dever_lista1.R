# LISTA 1
# Guilherme Braga

# Exercícios da matéria de Redes Neurais 1 do Professor Guilherme Rodrigues
# Mestrado em Estatística - Universidade de Brasília
# Matéria de Semestre 2024/01
# Feito no final de 2025 para estudo pessoal

# Os dados que eu anotei nos comentários sobre dados podem não estar corretos
# por conta de correções nessa implementação que eu fiz na Lista 2.
# Mas as análises estão ok.

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
 
X_teste <- as.matrix(dados_test[, c("x1","x2")])
y_teste <- dados_test$y

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
 
# wrapper par prever em um dataset X (matriz n×2)
pred_y_chapeu <- function(X, pesos, bias) {
  out <- lapply(1:nrow(X), function(i) predict_nn(X[i, ], pesos = pesos, bias = bias))
  sapply(out, `[[`, "y_chapeu")  # vetor numérico (n)
}
 
# Full Gradient Descent
for (t in 1:n_iter) {
  
  # gradiente no treino
  g <- grad_nn(pesos, bias, X_treino, y_treino)
  
  # atualização (descida)
  pesos <- pesos - eps * g$grad_pesos
  bias  <- bias  - eps * g$grad_bias
  
  # custos (para monitorar)
  hist_train[t] <- custo_nn(pesos, bias, X_treino, y_treino)
  
  # na nova implementação não precisa calcular isso
  # mas deixa
  hist_val[t]   <- custo_nn(pesos, bias, X_val, y_val)
  
  # melhor validação
  if (hist_val[t] < best_val) {
    best_val <- hist_val[t]
    best_iter <- t
    best_pesos <- pesos
    best_bias  <- bias
  }
  
  #print(t)
  # vou dar predict para pegar o y_chapeu
  best_y_hat_test <- pred_y_chapeu(X_teste, best_pesos, best_bias)
}

# dai aqui é os resultados da questão
best_val 
best_iter 
best_pesos 
best_bias 

length(best_y_hat_test) == nrow(X_teste)

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

df_test <- dados_test %>%
  mutate(
    y_hat = best_y_hat_test,
    resid = y - y_hat
  )

# valores previstos
ggplot(df_test, aes(x = x1, y = x2)) +
  geom_point(
    aes(colour = y_hat),
    size = 1.2,
    alpha = 0.35
  ) +
  scale_colour_gradient(
    low  = "white",
    high = "black",
    name = expression(hat(y))
  ) +
  coord_cartesian(expand = FALSE) +
  labs(
    x = expression(X[1]),
    y = expression(X[2]),
    title = "Valores previstos pela rede no conjunto de teste"
  )

# Interpretando
# Tendencia global no plano, inclinacão em X1
# O modelo aprendeu uma projecão suave mas não foi capaz de fazer a funcão original
# Era pra ser tipo, X1^3 −30 sin(X2)+10
# Não linear em X2 e cúbico em X1, não foi isso que vimos, foi linear

# resíduos
ggplot(df_test, aes(x = x1, y = x2)) +
  geom_point(
    aes(colour = resid),
    size = 1.2,
    alpha = 0.35
  ) +
  scale_colour_gradient2(
    midpoint = 0,
    low  = "blue",
    mid  = "white",
    high = "red",
    name = expression(y - hat(y))
  ) +
  coord_cartesian(expand = FALSE) +
  labs(
    x = expression(X[1]),
    y = expression(X[2]),
    title = "Resíduos da rede no conjunto de teste"
  )

# esse resíduo é estruturante
# não ta ai por acaso
# Parece um seno, mas era pra ser ruído mesmo
# Erro grande na rede

# Dei pro chatgpt analisar as imagens e ele deu esses insights

# Regiões com |X₂| grande:
#   a oscilação da função verdadeira é mais intensa, e a rede falha.

# Transições entre regimes do valor absoluto:
#   a rede não consegue representar bem o “vinco” da função.

# Bordas do domínio: maior erro sistemático.

# Isso confirma um viés estrutural na rede descrita
# me parece fazer sentido sim.

# -----------------------------------------------------

# ITEM H

# Botando o valor observado e o estimado no grafico, era pra ser uma curva bonita
# em que tudo coincide, mas na verdade tem uns picos nas estiacões, como se fosse um vício mesmo

ggplot(df_test, aes(x = y_hat, y = y)) +
  geom_point(alpha = 0.35, size = 1) +
  geom_abline(intercept = 0, slope = 1,
              colour = "red", linetype = "dashed", linewidth = 1) +
  labs(
    x = expression(hat(y)),
    y = "y observado",
    title = "Valores observados vs. valores previstos (teste)"
  )

# -----------------------------------------------------

# ITEM I

# Vou iterar 300 vezes e pegar o gradiente parcial w1 pra botar isso num grafico
# A cada loop, calcular a derivada parcial de 1 até o k daquele loop
# guardar no vetor e plotar

# theta fixo
pesos_fixos <- rep(0.1, 6)
bias_fixos  <- rep(0.1, 3)

# dados completos
X_full <- as.matrix(dados[, c("x1","x2")])
y_full <- dados$y

# vou iterar e pegar o gradiente de w1
K <- 300
grad_w1_k <- numeric(K)

for (k in 1:K) {
  
  # pegando todo mundo de 1 até k
  X_k <- X_full[1:k, , drop = FALSE]
  y_k <- y_full[1:k ]
  
  # calculo e salvo o valor calculado
  g_k <- grad_nn(pesos_fixos, bias_fixos, X_k, y_k)
  grad_w1_k[k] <- g_k$grad_pesos[1]  
  
}

# ai eu pego calculando uma vez tudo e esse era o baseline
g_full <- grad_nn(pesos_fixos, bias_fixos, X_full, y_full)
grad_w1_full <- g_full$grad_pesos[1]

plot(
  1:K, grad_w1_k,
  type = "l",
  lwd = 2,
  xlab = "Número de observações (k)",
  ylab = expression(partial(J)/partial(w[1])),
  main = expression("Convergência de w1, derivada parcial")
)
abline(h = grad_w1_full, col = "red", lwd = 2)

# comparaćão de tempo - chatgpt
mb <- microbenchmark(
  k300 = grad_nn(pesos_fixos, bias_fixos,
                 X_full[1:300, ],
                 y_full[1:300]),
  
  k100k = grad_nn(pesos_fixos, bias_fixos,
                  X_full,
                  y_full),
  
  times = 20
)

print(mb)

# Unit: microseconds
# expr      min        lq      mean    median       uq      max neval
# k300   44.980   47.2005   56.6452   52.4255   62.685    89.02    20
# k100k 3649.072 3782.1770 5269.5442 3865.5920 6587.758 11822.98    20


# k100k, que é o dataset inteiro é muuuuuito mais lento
# ou seja, a partir de algumas centenas de registros, temos a convergência esperada! E bem rápida!

# -----------------------------------------------------
# Modelos lineares para comparacão
# -----------------------------------------------------

# ITEM J

# Modelo Linear 1 é simplesmente uma regressão linear normal com x1 e x2 como covariáveis, ajustada no conjunto de treinamento

mod_lin1 <- lm(y ~ x1 + x2, data = dados_train)

summary(mod_lin1)

# nesse caso Yi = Beta Zero + Beta Um * X1 + Beta Dois * X2 + Erro
# y^ = 21,88 + 1,20 X1 − 4,25 X2

y_hat_lin1 <- predict(mod_lin1, newdata = dados_test)

mse_lin1 <- mean((dados_test$y - y_hat_lin1)^2)
mse_lin1
# Deu 134.5856

# Modelo Linear 2 é mais complicado que o primeiro
# Ele tem componentes quadrticas e um termo que x1 e x2 interagem

# Yi = Beta Zero + Beta Um * X1 + Beta Dois * X2 + Beta Tres * X1^2 + Beta Quatro * X2^2 + Beta Cinco (x1*x2) + Erro

mod_lin2 <- lm(y ~ x1 + x2 + I(x1^2) + I(x2^2) + I(x1*x2), data = dados_train)
summary(mod_lin2)

y_hat_lin2 <- predict(mod_lin2, newdata = dados_test)
mse_lin2 <- mean((dados_test$y - y_hat_lin2)^2)
mse_lin2
# Deu 93.62245 

# MSE da rede neural
# Soma de (Y teste - Y predito)^2 dividido por n
# Usando a função mean

mse_rna <- mean((dados_test$y - best_y_hat_test)^2)
mse_rna
# Deu 172.4924 

# O menor erro quadratico medio é o do modelo 2
# A rede neural foi incapaz de abstrair o modelo original nessa arquitetura
# O melhor modelo para previsão seria o modelo linear 2
# Apesar de que o modelo 1 já foi melhor do que a rede neural, sendo um modelo mais simples e tudo mais

# -----------------------------------------------------

# ITEM K
# Efeito da variação de X1

# meu modelo 1 é
# y^ = 21,88 + 1,20 X1 − 4,25 X2

# vou pegar todos os x1 de teste e adicionar 1
dados_test_x1_up <- dados_test
dados_test_x1_up$x1 <- dados_test_x1_up$x1 + 1

# MODELO 1
# fazer uma nova predição
y_hat_x1_up <- predict(mod_lin1, newdata = dados_test_x1_up)

# e comparar com a antiga
y_hat_base <- predict(mod_lin1, newdata = dados_test)

# vou dar summary nos dois resultados
summary(y_hat_x1_up)
summary(y_hat_base)

# na média, o resultado do y resultante quando eu adiciono uma unidade em x1 é de cerca de 1.2
# 23,018 - 21,815 ~= 1,2
# O que faz sentido dado a equação do modelo 1

# fazer uma nova predição
y_hat_x1_up <- predict(mod_lin2, newdata = dados_test_x1_up)

# e comparar com a antiga
y_hat_base <- predict(mod_lin2, newdata = dados_test)

# vou dar summary nos dois resultados
summary(y_hat_x1_up)
summary(y_hat_base)

# tá dando na mesma...

# -----------------------------------------------------

# ITEM L

# "Novamente, para cada um dos 3 modelos em estudo, calcule o percentual de vezes que o intervalo 
# de confiança de 95% (para uma nova observação!) capturou o valor de yi."


pred_ic_mod1 <- predict(
  mod_lin1,
  newdata = dados_test,
  interval = "prediction",
  level = 0.95
)

pred_ic_mod2 <- predict(
  mod_lin2,
  newdata = dados_test,
  interval = "prediction",
  level = 0.95
)

# é uma matriz com o resultado, o "upr" e o "lwr" "bounds"

inside_mod1 <- dados_test$y >= pred_ic_mod1[, "lwr"] &
  dados_test$y <= pred_ic_mod1[, "upr"]

inside_mod2 <- dados_test$y >= pred_ic_mod2[, "lwr"] &
  dados_test$y <= pred_ic_mod2[, "upr"]


proporcao_true_mod1 <- sum(inside_mod1) / length(inside_mod1)
proporcao_true_mod1
# 0,9588

proporcao_true_mod2 <- sum(inside_mod2) / length(inside_mod2)
proporcao_true_mod2
# 0,9668

# agora a rede neural
# sigma_chapeu = raiz quadrada do mse

sigma_hat <- sqrt(mse_rna)

z <- qnorm(0.975) 

# agora os bounds são essas variações dos resultados da rede
# df_test tem os resultado da rna

lwr <- df_test$y_hat - z * sigma_hat
upr <- df_test$y_hat + z * sigma_hat

inside_rna <- (df_test$y >= lwr) & (df_test$y <= upr)


proporcao_true_rna <- sum(inside_rna) / length(inside_rna)
proporcao_true_rna
# 0,9681

# -----------------------------------------------------

# ITEM M

# matriz de predições, modelo 1 nos dados de teste
pred_ic <- predict(
  mod_lin1,
  newdata = dados_test,
  interval = "confidence",
  level = 0.95
)

# disso sai o y predito, o "lwr" e o "upr". São os limites superiores e inferiores

# ai eu pego os intervalos do intervalo de confiança
# intervalo superior e inferior
inside_ic <- dados_test$y >= pred_ic[, "lwr"] &
  dados_test$y <= pred_ic[, "upr"]


# tipo o item H, mas eu vou ajustar os valores
plot(dados_test$x1, dados_test$x2,
     col = ifelse(inside_ic, "green", "red"),
     pch = 16,
     xlab = "x1",
     ylab = "x2",
     main = "Dispersão x1 vs x2 — Conjunto de Teste. Modelo Linear 1.")

legend("topright",
       legend = c("Dentro do IC (95%)", "Fora do IC (95%)"),
       col = c("green", "red"),
       pch = 16)


# Para um intervalo de confiança de 95%, a grande maioria das observações não foram satisfatórias.
# o intervalo de confiança avalia a incerteza da média condicional estimada e não a variabilidade total das observações individuais. 
# Mesmo com bom MSE, os intervalos de confiança são naturalmente estreitos e não englobam a maior parte dos valores observados.

# -----------------------------------------------------








