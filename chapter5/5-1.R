# 5.1 합성망 소개
# 컴퓨터 비전 애플리케이션에서 보편적으로 사용되는 딥러닝 모델의 일종
# layer_conv_2d와 layer_max_pooling_2d를 겹겹이 쌓은 형태

# 정리정돈
rm(list=ls())
path <- "C:/study/lecture/arbeit/keras-study"
setwd(path)

library(keras)

?layer_conv_2d
?layer_max_pooling_2d

# 소형 합성망 인스턴스화
# MNIST 데이터셋을 사용할 것이라, 28픽셀 * 28픽셀의 이미지를 처리할 수 있도록 input_shape 지정
model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

# 합성망 아키텍처 확인
# 소형 합성망인데도 훈련 가능한 파라미터 수 55744개
model

# 3D 텐서를 1D로 만들고, 조밀 계층 2개 추가
model <- model %>%
    layer_flatten() %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 10, activation = "softmax")

# 다시 아키텍처 확인
# 3개 계층이 추가되어 있음
# flatten 부분에서 출력 형태가 1D로 바뀌고, 마지막에는 0~9 사이의 숫자를 출력하도록 출력 형태가 10이 됨
model

# MNIST 데이터로 합성망 훈련하기
# 시간은 이전에 사용했던 모델 대비 훨씬 오래 걸린다...
mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images/255
test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images/255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
)

model %>% fit(
    train_images, 
    train_labels, 
    epochs = 5, 
    batch_size = 64
)

# 테스트 데이터로 모델 평가
# 정확도 무려 99.1%, 첫 예제로 MNIST 데이터 다뤘을 때 최종 평가 결과가 98.85%였으니 보다 향상된 결과
result <- model %>% evaluate(test_images, test_labels)
result

# 5.1.1 합성곱 연산
# 조밀 계층은 전역 패턴 학습, 합성곱 계층은 지역 패턴 학습
# 합성망이 학습하는 패턴은 변호나 불변성(Translation invariant)을 띄므로, 이미지의 어디에서나 패턴을 인식함
# 합성망은 패턴의 공간적 계층 구조(Spatial hierarchies)들을 학습할 수 있음, 점점 더 복잡하고 추상적인 시각적 개념을 효율적으로 학습
# 합성곱은 높이와 너비 공간 축, 깊이 채널 축이 있는 3D 텐서 특징 지도(Feature maps)를 통해 작동
# 합성곱 연산은 입력 특징 지도에서 조각들을 뽑아내 동일 변환을 적용, 3D 텐서인 출력 특징 지도(Output feature map) 생성
# 부분적 특징들이 모여 특정 완성 객체와 같은 고수준 개념으로 결합됨
# MNIST 예제의 경우, 첫 번째 합성곱 계층은 (28, 28, 1) 크기의 특징 지도로 (26, 26, 32) 크기의 특징 지도 출력
# 합성곱은 두 가지 주요 파라미터로 정의됨: 입력에서 추출한 조각의 크기(일반적으로 3*3 또는 5*5), 출력 특징 지도의 깊이(MNIST 예제에서는 32로 시작해 64로 끝남)
# 주요 파라미터는 계층으로 전달되는 첫 번째 인수, layer_conv_2d(출력깊이, c(창높이, 창너비))와 같은 식
# 보기 5.4 참고

