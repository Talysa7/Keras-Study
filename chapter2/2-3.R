# 2.3 신경망의 장비: 텐서 연산
# 이해하기 좀 어려웠다...

# 정리정돈
path <- "C:/study/lecture/keras-study"
setwd(path)
rm(list=ls())

# 계층을 서로 쌓아 망을 구축하는 예제에서, 계층의 사례
# 2D 텐서를 입력 받아 다른 2D 텐서를 반환하는 함수
# layer_dense(units = 512, activation = "relu")
# 구체적으로는 다음과 같음
# output = relu(dot(W, input) + b)
# ReLU 연산이 쓰임, relu(x)는 max(x, 0)이라고만 알아두자

# 모양이 다른 텐서 연산
# sweep() 함수를 사용해 더 높은 차원의 텐서와 더 낮은 차원의 텐서간 연산 수행 가능
# sweep(x, 2, y, '+')
# 두 번째 인자는 y를 sweep할 x의 크기를 지정한다는데, 예제로 보는 게 이해가 빠를 듯
# x는 (64, 3, 32, 10) 모양으로 된 난수 값 4D 텐서
x <- array(round(runif(1000,0,9)), dim = c(64, 3, 32, 10))
# y는 (32, 10) 모양의 5로 구성된 2D 텐서
y <- array(5, dim = c(32, 10))
# 하나는 4D 텐서, 하나는 2D 텐서니까 서로 모양이 다름
# 계산하면 어떻게 되는지 전후 비교
# 연산 전 x 첫 번째 행렬
head(x[1,1,,])
# sweep()로 pmax 연산하기 예제
# sweep할 x의 크기를 (3, 4) 행렬로 지정했는데...?
z <- sweep(x, c(3,4), y, pmax)
# 결과물은 x와 같은 모양
dim(z)
# x 첫 번째 행렬과 비교해보기, pmax 연산을 했으므로 5보다 큰 숫자만 있다.
head(z[1,1,,])
# 왜 두 번째 인자에 (3, 4) 행렬을 넘겼는지 생각해보기

# 내적 연산, 텐서 곱, 입력 텐서의 성분 결합
# 내적 연산에는 %*% 연산자 사용
# z <- x %*% y, 수학 표기법에서는 z = x . y
# 직접 돌려보지 않으면 이해가 안 되어서...
x <- array(round(runif(1000,0,9)), dim = c(5, 5))
y_1d <- c(1,2,3,4,5)
y_2d <- array(round(runif(1000,0,9)), dim = c(5, 5))
y_else <- array(round(runif(1000,0,9)), dim = c(5, 4))
x
y
# 두 벡터 x, y의 내적 연산 함수 예제
naive_vector_dot <- function(x, y) {
    z <- 0
    for (i in 1:length(x))
        z <- z + x[[i]] * y[[i]]
    z
}
# 원소 개수가 서로 같은 벡터만 내적 연산 호환
# 다음은 원소 개수가 서로 안 맞아 에러
naive_vector_dot(x, y_else)
# 다음은 행렬과 벡터 연산이라 에러
naive_vector_dot(x, y_1d)
# 행렬 x와 벡터 y의 내적 반환 예제
naive_matrix_vector_dot <- function(x, y) {
    z <- rep(0, nrow(x))
    for (i in 1:nrow(x))
        for (j in 1:ncol(x))
            z[[i]] <- z[[i]] + x[[i]] * y[[i]]
    z
}
# 위 함수로는 행렬과 벡터 내적 연산 가능
naive_matrix_vector_dot(x, y_1d)
# 이렇게도 할 수 있다는데
naive_matrix_vector_dot <- function(x, y) {
    z <- rep(0, nrow(x))
    for (i in 1:nrow(x))
        for (j in 1:ncol(x))
            z[[i]] <- naive_vector_dot(x[i,], y)
        z
}
# 연산 결과가 다르다?! 다른 계산 한 거였다...
naive_matrix_vector_dot(x, y_1d)
# 두 텐서 중 하나의 차원이 1차원 이상이면 %*%는 대칭 아님
# x %*% y와 y %*% x가 같지 않다는 말
# ncol(x) == nrow(y)인 경우에만 두 행렬간 내적 연산 가능, 결과는 (nrow(x), ncol(y)) 모양
naive_matrix_dot <- function(x, y) {
    z <- matrix(0, nrow = nrow(x), ncol = ncol(y))
    for (i in 1:nrow(x))
        for (j in 1:ncol(y)) {
            row_x <- x[i,]
            col_y <- y[,j]
            z[i,j] <- naive_vector_dot(row_x, col_y)
        }
    z
}
# 예제로 만든 (5,5) 행렬들로 예제 수행
naive_matrix_dot(x, y_2d)
# 벡터로 해 보면 에러 발생
naive_matrix_dot(x, y_1d)
naive_matrix_dot(y_1d, x)
# 48 페이지의 그림 보고 이해하기

# 새로운 예제 하기 위해 정리정돈
rm(list=ls())

# 텐서 모양 변경(Tensor reshaping)
# 행과 열을 표적 모양과 같도록 재조정하는 것

# dim <- () 함수가 아닌 array_reshape() 함수 사용
# 데이터가 재해석되도록 하기 위한 것으로 R의 기본인 열 중심 구문과 반대, Numpy나 Tensorflow 방식과 호환
# Keras에 전달할 R 배열을 다시 만들 때는 항상 array_reshape() 함수 사용할 것

# 예제용 행렬 만들기
# (3,2) 모양
x <- matrix(c(0, 1,
              2, 3,
              4, 5),
            nrow = 3, ncol = 2, byrow = TRUE)
x
dim(x)
# (6,1) 모양으로 바꾸기
x <- array_reshape(x, dim = c(6,1))
x
dim(x)
# 차이 비교, byrow TRUE라도 다르다!
dim_x <- matrix(x, nrow = 6, ncol = 1, byrow = TRUE)
dim_x
x[,1]
dim_x[,1]
# 다시 (2,3) 모양으로 바꾸기
x <- array_reshape(x, dim = c(2,3))
x
# 전치 행렬 만들기
x_t <- t(x)
x
x_t

# 50, 51페이지는 그래프 보고 이해할 것