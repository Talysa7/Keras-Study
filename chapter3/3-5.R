# 3.5 뉴스 분류: 다중 클래스 분류 예제
# 로이터 데이터셋 사용

# 정리정돈
rm(list=ls())
path <- "C:/study/lecture/arbeit/keras-study"
setwd(path)

# 3.5.1 로이터 데이터셋
# 1986년 로이터 통신이 발표, 46가지 주제의 단신, 주제마다 최소 10개 사례 포함
# 데이터셋에서 가장 자주 발생하는 1만 개의 단어로 데이터 제한
library(keras)
reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

# 훈련 데이터와 테스트 데이터 수 확인
# 훈련 데이터 8982개, 테스트 데이터 2246개
length(train_data)
length(test_data)

# 각 사례의 데이터 확인
# 하나의 데이터를 확인해보니 단어에 대한 인덱스인 정수로 된 목록
train_data[[1]]

# 각 주제를 나타내는 레이블(인덱스)은 0과 45 사이의 정수
train_labels[[1]]

# 현재 데이터는 단어 대신 해당 단어의 인덱스를 가지고 있지만 원래의 단어로 된 데이터를 보려면...
# word_index <- dataset_reuters_word_index()
# reverse_word_index <- names(word_index)
# names(reverse_word_index) <- word_index
# decoded_newswire <- sapply(train_data[[1]], function(index) {
#    word <- if(index >=3) reverse_word_index[[as.character(index - 3)]]
#    if (!is.null(word)) word else "?"
# })

# 3.5.2 데이터 준비
# 데이터 벡터화
Vectorize_sequences <- function(sequences, dimension = 10000) {
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    for (i in 1:length(sequences))
        results[i, sequences[[i]]] <- 1
    results
}

x_train <- Vectorize_sequences(train_data)
x_test <- Vectorize_sequences(test_data)

# 벡터화된 데이터 구조 확인
dim(x_train)
str(x_train)

# 레이블 벡터화
# 범주형 데이터에 적합한 원 핫 인코딩 사용
to_one_hot <- function(labels, dimension = 46) {
    results <- matrix(0, nrow = length(labels), ncol = dimension)
    for (i in 1:length(labels))
        results[i, labels[[i]] + 1] <- 1
    results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

# 케라스에 내장된 방법을 사용하려면 categorical() 사용
# one_hot_train_labels <- to_categorical(train_labels)
# one_hot_test_labels <- to_categorical(test_labels)

# 벡터화된 레이블 구조 확인
dim(one_hot_train_labels)
str(one_hot_train_labels)

# 3.5.3 망 구축
# 각 계층은 이전 계층의 출력에 있는 정보에만 액세스할 수 있음
# 이전 계층에서 분류에 필요한 일부 정보가 삭제되면 이후 계층에서 복구 불가능
# 각 계층에서 잠재적으로 정보 병목 현상이 발생할 수 있어 필요한 데이터가 영구히 사라지는 문제에 대비해 더 큰 계층 사용
# softmax 활성 함수는 서로 다른 46개 클래스에 대한 확률 분포를 출력, 각 클래스에 속할 확률을 출력하며 합은 1
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

# 모델 컴파일하기
# 손실 함수, 계량 함수, 최적화기 지정
# 손실 함수로는 categorical_crossentropy 사용, 망에 의한 확률 분포와 레이블의 실제 분포 사이 거리를 측정, 최소화
model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

# 3.5.4 접근법 검증하기
# 훈련 데이터 일부를 검증 집합으로 돌려보기
val_indices <- 1:1000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- one_hot_train_labels[val_indices,]
partial_y_train <- one_hot_train_labels[-val_indices,]

# 모델 훈련하기
history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 9 에포크 이후 과적합이 발생하므로 새로운 망을 다시 9 에포크 훈련하고 테스트 집합으로 평가
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 9,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 결과 확인, 정확도 79%
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 무작위로 실제 값과 비교해보기, 약 18%
# test_labels_copy <- test_labels
# test_labels_copy <- sample(test_labels_copy)
# length(which(test_labels == test_labels_copy)) / length(test_labels)

# 3.5.5 새 데이터에 대한 예측 생성
# predict()로 전체 주제 확률 분포 반환 여부를 확인할 수 있음
predictions <- model %>% predict(x_test)

# 예측은 테스트 데이터 수와 같고 길이 46인 벡터
# 확률 분포이기 때문에 벡터 합의 계수는 1
# 가장 값이 큰 클래스가 해당 데이터가 속한 주제일 가능성이 높은 클래스
dim(predictions)
sum(predictions[1,])
which.max(predictions[1,])

# 3.5.6 레이블과 손실을 처리하는 다른 방법
# 손실 함수로 categorical_crossentropy를 사용했는데, 이 손실함수는 레이블이 범주형 부호화를 따르기를 기대함
# 정수 레이블이라면 sparse_categorical_crossentropy를 사용해야 함

# 3.5.7 충분히 큰 중간 계층을 갖는 것의 중요성
# 최종 출력물이 46차원이기 때문에, 은닉 유닛이 46개 미만인 중간 계층은 피해야 함
# 46개 미만이 중간계층을 사용하면 어떤 일이 발생하는지 확인해보기
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 정확도 68%로 이전 결과(79%) 대비 매우 낮다
# 너무 적은 중간 공간에 많은 정보를 압축하려다보니 일어나는 현상
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 3.5.8 추가 실험
# 더 작은 계층이나 큰 계층 사용해보기

# 더 작은 계층 사용해보기
model <- keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 결과 확인, 시간은 약간 더 적게 걸렸고 정확도는 78%로 비슷, 손실은 조금 더 큼
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 더 많은 계층 사용해보기
model <- keras_model_sequential() %>%
    layer_dense(units = 128, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 결과 확인, 손실은 더 늘었고 정확도는 78%로 비슷, 처리 시간 아주 약간 더 걸림, 결국 효율적이지 않은 것으로 판단됨
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 은닉 계층 더 많이 또는 적게 사용해보기

# 하나의 은닉 계층만 사용해보기
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 결과 확인, 정확도 약 80%이고 손실도 크게 차이 나지 않음, 소요 시간 비슷
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 세 개의 은닉 계층 사용해보기
model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dense(units = 46, activation = "softmax")

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

history <- model %>% fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size = 512,
    validation_data = list(x_val, y_val)
)

# 결과 확인, 정확도는 76%로 더 낮고 손실도 더 큼, 시간은 비슷하게 소요됨
results <- model %>% evaluate(x_test, one_hot_test_labels)
results

# 은닉 계층의 수가 적거나 많은게 무조건적으로 좋고 나쁜 것은 아닌 듯...

# 3.5.9 결론
# N개 클래스에 해당하는 데이터들을 분류하려면 N개 클래스의 출력으로 끝나야 함
# 확률 분포를 다룰 때, 최종 계층의 활성 함수는 softmax
# categorical crossentropy는 범주형 문제에 사용하는 손실 함수
# 레이블 처리는 원 핫 인코딩 또는 정수 부호화
# 중간 계층이 너무 작으면 정보 병목 현상이 일어날 수 있음