# 3.6 주택 가격 예측: 회귀 예제

# 정리정돈
rm(list=ls())
path <- "C:/study/lecture/arbeit/keras-study"
setwd(path)

# 3.6.1 보스턴 주택 가격 데이터 셋
# 1970년대 중반 보스턴 시 교외 지역의 주택 평균 가격 예측
# 척도가 서로 다른 특징들을 가진 506개의 데이터

# 데이터셋 적재하기
library(keras)
dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

# 데이터셋 확인
# 13가지 특징이 있는 404개의 훈련 표본과 102개의 테스트 표본
# 표적(Target)은 소유자가 거주하는 주택의 중위수 값, 천 달러($) 단위
str(train_data)
str(train_targets)
str(test_data)

# 3.6.2 데이터 준비
# 특징별로 데이터 정규화하기
# 각 특징에서 평균을 뺀 후 표준편차로 나눠 0을 중심으로 단위 표준편차를 갖게 하는 식
# scale() 함수로 R에서 쉽게 처리 가능

# 훈련 데이터의 평균 및 표준편차 계산
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)

# 훈련 데이터의 평균 및 표준편차를 사용해 훈련 데이터 및 테스트 데이터의 크기 조정
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)

# 데이터 정규화 결과 확인
head(train_data)

# 3.6.3 망 구축
# 이전 예제까지는 매번 모델을 정의하고 컴파일했는데, 이번엔 일련의 작업을 하나의 함수로 묶었음
build_model <- function() {
    model <- keras_model_sequential() %>%
        # dim(train_data)[[2]]으로 한 이유는 13가지 특징을 가지고 학습하기 때문
        layer_dense(units = 64, activation = "relu", input_shape = dim(train_data)[[2]]) %>%
        layer_dense(units = 64, activation = "relu") %>%
        # 예측 가격, 즉 타겟 하나만 출력하면 되기 때문에 units = 1, 활성 함수 없는 단일 유닛으로 마무리
        # 연속적인 값을 예측하여 일종의 선형 계층이 된다
        layer_dense(units =1)
    
    model %>% compile(
        optimizer = "rmsprop",
        # 평균제곱오차를 손실 함수로 사용
        loss = "mse",
        # 평균절대오차로 계량 함수 사용, 예측과 표적간 사이의 절댓값, 예측이 평균적으로 얼마나 빗겨나갈지 알려줌
        metrics = c("mae")
    )
}

# 3.6.4 k겹 검증을 사용해 접근 방식 검증하기
# k겹 교차 검증(k-fold cross validation)을 사용해 모델 평가
# 검증 집합이 작아질 수 있고, 검증 분할 관련해 높은 분산이 있을 수 있는 경우에 유용
# 어떤 한 부분을 평가하는 동안 나머지 부분에서 각각 훈련하는 식으로 구성
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE)

num_epochs <- 100
all_scores <- c()

# 시간이 꽤나 걸린다...
for (i in 1:k) {
    # 콘솔에 진행 상황 파악 용도로 출력하는 문구
    cat("processing fold #", i, "\n")
    
    # 검증 데이터 준비: k 부분 데이터 준비
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    # 훈련 데이터 준비: 나머지 모든 데이터
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    # 모델 빌드 및 훈련
    model <- build_model()
    model %>% fit(partial_train_data, partial_train_targets, epochs = num_epochs, batch_size = 1, verbose = 0)
    
    # 검증 데이터로 모델 평가
    # 내 PC에서 실행 시 "$ operator is invalid for atomic vectors"가 발생해, 값에 접근하는 방식을 수정함
    # 교재에는 mean_absolute_error인데, dim()으로 확인 시 mae여서 수정
    results <- model %>% evaluate(val_data, val_targets, verbose = 0)
    all_scores <- c(all_scores, results["mae"])
}

# 결과 확인, 2376달러나 오차가 난다
# 주택 가격대가 1만~5만이므로 오차가 상당히 큰 편
all_scores
mean(all_scores)

# 오차를 개선하기 위해 망을 300 에포크만큼 훈련하고, 진행 상황을 확인할 수 있도록 수정
# 교재에는 500 에포크였으나, 300 에포크도 상당히 오래 걸리고, 125 에포크부터 과적합이 일어난다는 사실을 교재에서 알았으므로 300만 수행
# 실제로는 이럴 수 없겠지...
num_epochs <- 300
all_mae_histories <- NULL

# 300 에포크라 상당히 오래걸린다...
for (i in 1:k) {
    # 콘솔에 진행 상황 파악 용도로 출력하는 문구
    cat("processing fold #", i, "\n")
    
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    # 모델 빌드
    model <- build_model()
    
    # history() 사용
    history <- model %>% fit(
        partial_train_data, 
        partial_train_targets, 
        validation_data = list(val_data, val_targets), 
        epochs = num_epochs, 
        batch_size = 1, 
        verbose = 0
    )
    
    # 교재에는 val_mean_absolute_error인데, dim()으로 확인 시 val_mae여서 수정
    mae_history <- history$metrics$val_mae
    all_mae_histories <- rbind(all_mae_histories, mae_history)
}

# 처음에 시도했을 때는 오류가 있었음
# warnings()로 오류를 확인할 수 있는데, 이것만으로 정확히 파악은 어렵다
# average_mae_history의 길이가 0이어서 해당 부분에 문제가 있는 것으로 보고 수정 후 재시도함
# warnings()

# 모든 겹에 대한 에포크당 MAE 평균 계산
average_mae_history <- data.frame(
    epoch = seq(1:ncol(all_mae_histories)),
    validation_mae = apply(all_mae_histories, 2, mean)
)

# 시각화
library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

# 위 표는 척도 문제로 보기 어려우므로 다음과 같이 출력해본다
# 약 80 에포크 이후 큰 개선 없음을 알 수 있음
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# 최종 모델 훈련하기
model <- build_model()
model %>% fit(train_data, train_targets, epochs = 80, batch_size = 16, verbose = 0)
result <- model %>% evaluate(test_data, test_targets)

# 최종 결과 확인, 평균적으로 무려 2826달러나 오차가 있을 수 있음
# 추가 개선 필요
result

# 3.6.5 결론
# 회귀는 분류로 수행되지 않고, 서로 다른 손실 함수를 사용
# 평균제곱오차(MAE)는 회귀에 일반적으로 사용되는 손실 함수
# 입력 데이터의 특징이 범위가 다른 값을 가지면, 전처리 단계에서 조정해야 함(Scaling)
# 사용할 수 있는 데이터가 거의 없는 경우 k겹 검증을 사용해 모델을 보다 신뢰할 수 있게 할 것
# 훈련 데이터가 거의 없는 경우, 심각한 과적합을 피하기 위해 은닉 계층이 거의 없는 소규모 망을 사용하는 게 바람직