# 5.3 사전 훈련 합성망 사용하기
# 대용량 데이터셋의 대규모 이미지 분류 작업을 통해 훈련하고 저장된 망을 또다른 컴퓨터 비전 문제에 활용
# 이미지넷 데이터셋으로 훈련된 합성망을 사용해 개 고양이 분류하기
# VGG 아키텍처 사용해 특징 추출(Feature extraction), 미세 조정(fine-tuning) 학습

# 5.3.1 특징 추출
# 이전에 훈련된 합성곱 기반과 새로운 데이터를 위한 조밀한 분류기 사용
# 분류기는 재사용하지 않아야 함, 일반화 측면에서 거의 쓸모가 없음

# VGG 망의 합성곱 기반을 사용해 개 고양이 이미지의 특징 추출 후 분류기 훈련시키기
# VGG 모델은 케라스에서 제공
library(keras)

# 3가지 인자 전달해야 함, 데이터 다운로드에 약간 시간 걸림
conv_base <- application_vgg16(
  weights = "imagenet", # 가중치 검사점
  include_top = FALSE, # 분류기 포함 여부
  input_shape = c(150, 150, 3) # 입력 텐서 모양
)

# 아키텍처 확인, 최종 특징 지도는 (4, 4, 512) 모양
conv_base

# 조밀 연결 분류기를 부착하자!
# 방법은 두 가지
# 1. 데이터셋에 대한 합성곱 기반 실행, 출력을 디스크에 한 개 배열로 기록, 조밀 연결 분류기 입력으로 사용
# 이 방법을 쓰면 합성곱 기반을 한 번만 실행해 계산은 빠르지만, 데이터 보강 기술을 사용할 수 없음
# 2. 상단에 조밀 계층을 추가하고 입력 데이터에서 끝까지 모든 것을 실행해 모델 확장
# 데이터 보강이 가능하지만 비싸다. GPU 없으면...

# 먼저 데이터 보강 없이 빠르게 특징 추출하는 방법부터 학습
# image_data_generator 인스턴스를 실행해 이미지를 배열 및 레이블로 추출
# 모델에서 predict 메서드를 호출해 이미지에서 특징 추출
base_dir <- "C:/study/lecture/arbeit/cat-and-dog/my"
train_dir <- "C:/study/lecture/arbeit/cat-and-dog/my/train"
validation_dir <- "C:/study/lecture/arbeit/cat-and-dog/my/validation"
test_dir <- "C:/study/lecture/arbeit/cat-and-dog/my/test"

datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20

extract_features <- function(directory, sample_count) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch) # 이미지에서 특징 추출
    
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      break
  }
  
  list(
    features = features,
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

# 추출된 특징들은 (표본들, 4, 4, 512) 모양이므로 평평하게 만들기
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# 분류기 정의 및 훈련
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features,
  train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

# 결과 표시
# 드롭아웃 비율 0.5에 정확도는 90%이지만 시작부터 과적합... 작은 이미지 데이터셋만 사용했기 때문
plot(history)

# 데이터 보강을 사용한 특징 추출
# GPU를 사용할 수 있어야... 과연 돌아갈까?
# 앞서 가져온 conv_base 모델을 확장, 계층을 추가하듯이...
model <- keras_model_sequential() %>%
  conv_base %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

# 아키텍처를 확인해보면 conv_base에 추가한 계층들이!
model

# 모델 컴파일&훈련 전 합성곱 기반 동결하기
# 이전에 학습된 표현이 훈련 중에 수정되지 않도록?
# 이 설정을 사용해 추가한 두 개 조밀 계층의 가중치만 훈련됨
# 총 4개의 가중치 텐서
cat("This is the number of trainable weights before freezing",
    "the conv base:", length(model$trainable_weights), "\n")
freeze_weights(conv_base)
cat("This is the number of trainable weights after freezing",
    "the conv base:", length(model$trainable_weights), "\n")

# 변경사항을 적용하려면 모델 컴파일, 컴파일 후에 가중치 조절 기능을 수정하면 모델 다시 컴파일 해야 함
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)

# 과연 잘 돌아갈까... 다행히 돌아가기는 하는데 시간이 제법 걸린다. 30 에포크라 다행..?
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# 결과 재확인, 정확도는 90%이지만 과적합 측면을 고려하면 첫 번째 방법보다 나음
plot(history)

# 5.3.2 미세 조정
# 훈련된 기본 망 위에 맞춤 망 추가 -> 기본 망 동결 -> 추가한 부분 훈련 -> 기본 망 일부 동결 해제 -> 함께 훈련

# 동결 해제
unfreeze_weights(conv_base, from = "block3_conv1")

# 미세 조정, RMSprop 최적화기로 학습 속도를 느리게 해 수정하는 정도를 제한
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)

# 오래 걸린다... 내 PC로는 한 에포크당 9분 남짓.
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch= 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

# 결과 확인
plot(history)

# 테스트 데이터에서 최종 평가
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

# 손실, 정확도
model %>% evaluate_generator(test_generator, steps = 50)

# 모델 저장(오래 걸렸기 때문에 저장해둔다!)
model %>% save_model_hdf5("cats_and_dogs_small_3")