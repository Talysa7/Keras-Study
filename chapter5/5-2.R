# 5.2 소규모 데이터셋을 이용해 합성망을 처음부터 훈련하기
# 캐글 개/고양이 데이터셋 사용

# 정리정돈
rm(list=ls())
path <- "C:/study/lecture/arbeit/keras-study"
setwd(path)

# 5.2.1 소규모 데이터 문제와 딥러닝의 연관성
# 딥러닝은 훈련 데이터의 흥미로운 특징들을 스스로 찾을 수 있음
# 모델이 작고 규칙적이며 작업이 간단하다면 몇백 개 정도의 데이터로도 충분한 경우 있음

# 5.2.2 데이터 내려받기
# 개/고양이 데이터셋 사용, 2000개의 학습 데이터, 1000개의 검증 데이터, 1000개의 테스트 데이터로 분류
# 다운로드 받은 데이터 경로별로 분류
original_dataset_dir <- "C:/study/lecture/arbeit/cat-and-dog/train"

base_dir <- "C:/study/lecture/arbeit/cat-and-dog/my"
dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)
train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)
validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)
test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_cats_dir))

fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_cats_dir))

fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(train_dogs_dir))

fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(validation_dogs_dir))

fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, fnames), 
          file.path(test_dogs_dir))

# 온전한지 검사, 각각 몇 장의 사진 포함하고 있는지 계산
cat("total training cat images:", length(list.files(train_cats_dir)), "\n")
cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")
cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")
cat("total validation dog images:", length(list.files(validation_dogs_dir)), "\n")
cat("total test cat images:", length(list.files(test_cats_dir)), "\n")
cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")

# 5.2.3 망구축
# 개/고양이 분류를 위한 소형 합성망 인스턴스화하기
library(keras)
model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

summary(model)

# 시그모이드 유닛으로 망을 마무리했으므로 이진 교차 엔트로피를 손실 함수로 사용, 최적화기는 RMSProp
model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
)

# 5.2.4 데이터 전처리
# 그림 파일 읽기 -> RGB 격자로 부호화 -> 텐서로 변환 -> 0~1 구간 값으로 재조정
# 케라스에서 제공하는 image_data_generator() 사용
train_datagen <- image_data_generator(rescale = 1/255) # 0~1 구간 값으로 조정
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
    train_dir,
    train_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
    validation_dir,
    validation_datagen,
    target_size = c(150, 150),
    batch_size = 20,
    class_mode = "binary"
)

# 생성기를 사용해 모델에 데이터 적합시키기
# fit_generator 함수 사용, steps_per_epoch 인수에 가져올 표본 수 전달
history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 50
)

# 모델 저장하기
model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

# 훈련 중 손실 및 정확도 곡선 표시하기
plot(history)

# 훈련 표본이 적어 과적합 우려
# 드롭아웃, 가중치 감소 등 활용한 과적합 완화 필요

# image_data_generator를 통한 데이터 보강 설정 구성하기 예제
datagen <- image_data_generator(
    rescale = 1/255,
    rotation_range = 40, # 각도 단위 값, 그림을 임의로 회전
    width_shift_range = 0.2, # 그림을 가로 또는 세로로 임의 변환하는 범위
    height_shift_range = 0.2,
    shear_range = 0.2, # 임의로 가위질 변환 적용
    zoom_range = 0.2, # 그림을 무작위로 확대
    horizontal_flip = TRUE, # 수평 비대칭을 가정하지 않은 경우, 관련 이미지의 절반을 무작위 반전
    fill_mode = "nearest" # 새로 생성된 픽셀을 채우는 방법
)

# 어떤 식으로 이미지가 보강되는지 확인
fnames <- list.files(train_cats_dir, full.names = TRUE)
img_path <- fnames[[3]] # 이미지 하나 선택

img <- image_load(img_path, target_size = c(150, 150)) # 이미지 읽어 크기 조정
img_array <- image_to_array(img) # 이미지를 (150, 150, 3) 배열로 변환
img_array <- array_reshape(img_array, c(1, 150, 150, 3)) # 배열을 (1, 150, 150, 3)으로 변경

augmentation_generator <- flow_images_from_data(
    img_array,
    generator = datagen,
    batch_size = 1
)

op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))
for (i in 1:4) {
    atch <- generator_next(augmentation_generator)
    plot(as.raster(batch[1,,,]))
}

par(op)

# 위와 같은 데이터 보강 구성을 사용해 새 망을 훈련하면 망에 동일한 입력이 두 번 표시되지 않음
# 새로운 정보는 생성할 수 없고, 기존 정보를 다시 섞어 쓰는 것, 과적합을 없애기에 충분치 않음
# 과적합에 더 대응하려면 분류기 바로 앞 모델에 드롭아웃 계층 추가

# 드롭아웃을 포함하는 새로운 합성망 정의하기
model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("acc")
)

# 새로운 망으로 합성망 훈련하기
test_datagen <- image_data_generator(rescale = 1/255)
train_datagen <- flow_images_from_directory(
    train_dir,
    datagen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)
validation_datagen <- flow_images_from_directory(
    validation_dir,
    test_datagen,
    target_size = c(150, 150),
    batch_size = 32,
    class_mode = "binary"
)

history <- model %>% fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50
)

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")

# 데이터 보강 및 드롭아웃 후 과적합되지 않고 상대적으로 향상된 정확도(82%)에 도달
# 최대 87%까지 정확도를 높일 수 있지만, 더 높은 수준은 어려움
# 사전 훈련 모델을 사용해 정확도를 더 높일 수 있음