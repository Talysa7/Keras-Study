# 5.4 합성망이 학습한 내용 시각화하기
# 이미지 학습을 시키면서, 합성망을 지나는 동안 이미지를 어떻게 변환하고 특징을 파악하는지 눈으로 보기

# 정리정돈
rm(list=ls())
path <- "C:/study/lecture/arbeit/keras-study"
setwd(path)


# 기존에 저장해둔 모델 적재해 사용
library(keras)

model <- load_model_hdf5("cats_and_dogs_small_2.h5")
model

# 고양이 이미지 한 장 전처리하기
img_path <- "C:/study/lecture/arbeit/cat-and-dog/my/test/cats/cat.1700.jpg"
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3))
img_tensor <- img_tensor/255

# 테스트용 고양이 이미지 출력하기
plot(as.raster(img_tensor[1,,,]))

# 테스트용 고양이 이미지가 모델을 통해 변화하는 모습 시각화하여 관찰하기
# 상위 8개 계층의 출력을 추출해 반환하는 모델 만들기
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, output = layer_outputs)

# 예측 모드에서 모델 실행
activations <- activation_model %>% predict(img_tensor)

# 첫번째 합성곱 계층의 활성화, 특징 지도의 크기는 원래 이미지의 크기와 다름
first_layer_activation <- activations[[1]]
dim(first_layer_activation)

# 위 특징 지도 그려보기
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(channel), axes = FALSE, asp = 1, col = terrain.colors(12))
}

# 특징 지도 중 첫 번째 채널 그리기, 원래 이미지와 큰 차이 없어 보임
plot_channel(first_layer_activation[1,,,1])

# 두 번째 채널, 첫 번째 채널과 다른 부분이 눈에 띔, 눈에 드러남
plot_channel(first_layer_activation[1,,,2])

# 일곱 번째 채널, 마찬가지로 다름, 털결이 다른 부분이 드러남
plot_channel(first_layer_activation[1,,,7])

# 모든 채널을 추출해 나란히 그려놓기
image_size <- 58
images_per_row <- 16

for (i in 1:8) {
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0("cat_activations_", i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for(col in 0:(n_cols - 1)) {
    for(row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  
  par(op)
  dev.off()
}

# 위 반복문을 통해 만들어진 이미지는 파일로 저장되어 있음
# 고양이는 점점 추상적인 표현으로 나타나 나중에는 고양이였는지 알아볼 수 없음
# 현대 미술 같이 변한다...
# 정보는 반복적으로 변환되며 필요 없는 정보를 걸러내고 고수준의 시각적인 개념으로 바뀜