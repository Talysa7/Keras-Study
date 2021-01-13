# 2.1 신경망 둘러보기

# 정리정돈
path <- "C:/study/lecture/keras-study"
setwd(path)
rm(list=ls())

# Keras 설치
install.packages("keras")
library(keras)

# 책에는 안 써있지만 Miniconda도 설치해야 하네...
# 당연하지만 Tensorflow도 설치해야 하고.
# Miniconda는 예제 실행했더니 설치할 거냐고 물어봐서 설치했음.
# Tensorflow 설치는 Stackoverflow에서 찾음.
# https://github.com/rstudio/tensorflow/blob/master/README.md
library(tensorflow)
install_tensorflow()

# 이제 다음 예제를 실행할 수 있다!
# MNIST 데이터셋. Hello World 같은 것. 손글씨로 쓴 글자가 0~9 중 무엇인지 분류하는 모델을 학습한다.
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# 데이터 구조 확인
str(train_images)
str(train_labels)
str(test_images)
str(test_labels)

# 망 구축, 지금은 잘 이해되지 않는다.
# 한 가지 모델로 시작해서 다른 계층을 추가하는 방식
# 계층(Layer)은 데이터 필터와 같은 데이터 처리 모듈
# softmax 계층은 합해서 1이 되는 확률 점수 배열 반환, 각 점수는 현재 자릿수 이미지가 10자리 클래스 중 하나에 속할 확률
# 가장 점수가 높은 클래스의 숫자가 그 글자의 숫자겠지?
network <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = "relu", input_shape = c(28*28)) %>%
    layer_dense(units = 10, activation = "softmax")

# 망을 훈련하려면 다음 세 가지를 선택
# 1. 손실 함수(loss function)
# 2. 최적화기(optimizer)
# 3. 계량(metrics), 이번에는 정확도
# compile()은 망을 새로운 망 객체를 반환하는 대신 망을 수정
network %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

# 모든 값이 [0, 1] 구간이 되도록 데이터 전처리
# /255는 뭔지 모르겠다... RGB?
# dim() 말고 array_reshape() 사용, 이유는 나중에 설명해 준다고...
train_images <- array_reshape(train_images, c(60000, 28*28))
train_images <- train_images/255
test_images <- array_reshape(test_images, c(10000, 28*28))
test_images <- test_images/255

# 궁금해서 구조를 확인해보니 뭔가 바뀌었다!
# 값이 0 or 1
str(train_images)
str(train_labels)
str(test_images)
str(test_labels)


# 레이블 범주화, 이것도 나중에 설명해 준다고...
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# 궁금하니까 이것도 뭔가 찍어본다.
# 마찬가지로 값이 0 or 1
str(train_labels)
str(test_labels)
head(train_labels)
head(test_labels)

# 모델을 훈련 데이터에 적합하게 맞추기(fit)
# 시간이 약간 걸리고 이미지로 결과를 보여준다. 신기...
# 갈수록 손실(loss)은 감소, 정확도(accuracy)는 증가
# 최종 정확도는 98.85%! 헐...
network %>% fit(train_images, train_labels, epochs = 5, batch_size = 128)

# 모델이 테스트 집합에서 잘 수행되는지 평가
# 손실은 7.18%, 정확도는 97.92%로 훈련 집합보다 낮다
# 과적합의 예라고... 머신러닝 모델은 새 데이터에서 정확도가 떨어지는 경향이 있단다. 나중에 설명해 준다고.
metrics <- network %>% evaluate(test_images, test_labels)
metrics

# 10개의 표본 예측 생성하기
# 테스트 이미지 배열에서 열 개 예측시켜보니 7, 2, 1, 0, 4, 1, 4, 9, 5, 9...
network %>% predict_classes(test_images[1:10,])