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

# 5.2.4 합성망 필터 시각화
# 빈 입력 이미지에서 시작, 합성곱의 입력 이미지 값에 경사 하강 적용, 최대로 반응하는 입력 이미지 도출
# 주어진 필터이 값을 최대화하는 손실함수, 확률적 경사하강을 사용

# 손실 텐서 정의
model <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE
)

layer_name <- "block3_conv1"
filter_index <- 1

layer_output <- model$get_layer(layer_name)$output
loss <- k_mean(layer_output[,,,filter_index])

# 경사 하강 구현, 에러가 나기 때문에 eager execution을 비활성화함
# RuntimeError: tf.gradients is not supported when eager execution is enabled. Use tf.GradientTape instead. 
library(tensorflow)
tf$compat$v1$disable_eager_execution()
grads <- k_gradients(loss, model$input)[[1]]

# 경사 정규화
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)

# 손실 텐서 및 경사 텐서의 값을 계산하는 함수 정의
iterate <- k_function(list(model$input), list(loss, grads))

c(loss_value, grads_value) %<-%
  iterate(list(array(0,dim = c(1, 150, 150, 3))))

# 노이즈 있는 회색 이미지로 시작
input_img_data <-
  array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128

step <-1
for (i in 1:40) {
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  input_img_data <- input_img_data + (grads_value * step)
}

# 결과로 나오는 이미지 텐서는 [0, 255] 내의 (1, 150, 150, 3) 모양의 부동소수점 값을 가진 텐서
# 이 텐서를 후처리해 표시 가능한 이미지로 만들기
deprocess_image <- function(x) {
  dms <- dim(x)
  
  # 텐서 정규화, 0을 중심으로 std 0.1로 만듬
  x <- x - mean(x)
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1
  
  # [0, 1]에 맞춰 잘라내기
  x <- x + 0.5
  x <- pmax(0, pmin(x, 1))
  
  # 원본 이미지 크기를 반환
  array(x, dim = dms)
}

# 계층 이름과 필터 인덱스를 입력으로 사용해
#지정된 필터의 활성을 최대화하는 패턴을 나타내는 유효한 이미지 텐서를 반환하는 R 함수
generate_pattern <- function(layer_name, filter_index, size = 150) {
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index])
  
  grads <- k_gradients(loss, model$input)[[1]]
  
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  iterate <- k_function(list(model$input), list(loss, grads))
  
  input_img_data <- array(runif(size * size * 3), dim = c(1, 150, 150, 3)) * 20 + 128
  
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step)
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img)
}

# 시각화, 첫 번째 채널이 최대로 응답하는 패턴 보기
# block3_conv1 계층의 필터 1은 도트 패턴에 반응하는 것처럼 보임
library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

# 모든 계층의 모든 필터 시각화 가능
# 각 계층의 처음 필터 64개, 각 합성곱 블록의 첫 번째 계층 확인해보기

install.packages("gridExtra")
library(gridExtra)

dir.create("vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1")) {
  size <- 140
  png(paste0("vgg_filters/", layer_name, ".png"), width = 8 * size, height = 8 * size)
  
  grobs <- list()
  for (i in 0:7) {
    for (j in 0:7) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern, width = unit(0.9, "npc"), height = unit(0.9, "npc"))
      
      grobs[[length(grobs)+1]] <- grob
    }
  }
  
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}

# 이 시각화된 필터들은 합성망 계층이 세상을 보는 방법을 설명함
# 합성망의 각 계층은 입력을 필터의 조합으로 표현할 수 있도록 필터 모음을 학습
# 질감, 윤곽선, 색상 같은 것들이 부호화됨

# 5.4.3 클래스 활성의 열 지도 시각화
# 이미지의 어떤 부분이 특정 클래스를 활성화하는지?

model <- application_vgg16(weights = "imagenet")

# 위 모델은 224*224 크기의 이미지들로 훈련되었으므로, 사용하려는 이미지의 전처리 필요
img_path <- "C:/study/lecture/arbeit/keras-study/chapter5/creative_commons_elephant.jpg"
img <- image_load(img_path, target_size = c(224, 224)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 224, 224, 3)) %>%
  imagenet_preprocess_input()

# 예측 벡터를 사람이 읽을 수 있도록 복호화
# 아프리카 코끼리일 가능성의 점수가 높게 나왔음
# 이 이미지를 통해 최대로 활성화된 에측 벡터의 항목은 인덱스 387번 아프리카 코끼리인 것
preds <- model%>%predict(img)
imagenet_decode_predictions(preds, top = 3)
which.max(preds[1,])

# 아프리카 코끼리와 같은 이미지 부분을 시각화하기 위한 Grad-CAM 과정 설정

# 예측 벡터의 아프리카 코끼리 항목
african_elephant_output <- model$output[,387]

# 마지막 합성곱 계층인 "block5_conv3"의 출력 특징 지도
last_conv_layer <- model %>% get_layer("block5_conv3")

# 위 출력 특징 지도와 관련된 아프리카 코끼리 클래스의 경사
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]]

# 경사의 평균 강도
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

# 표본 이미지 한 장이 주어졌을 때, pooled_grads, block5_conv3 출력 특징 지도 출력
iterate <- k_function(list(model$input), list(pooled_grads, last_conv_layer$output[1,,,]))

# 두 코끼리의 견본 이미지가 주어졌을 때의 두 값
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

# 채널의 중요도를 각 채널에 곱하기
for (i in 1:512) {
  conv_layer_output_value[,,i] <- conv_layer_output_value[,,i] * pooled_grads_value[[i]]
}

# 클래스 활성화 열 지도
heatmap <- apply(conv_layer_output_value, c(1, 2), mean)

# 시각화를 위한 정규화
heatmap <- pmax(heatmap, 0)
heatmap <- heatmap/max(heatmap)

write_heatmap <- function(heatmap, filename, width = 224, height = 224, bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0, 0, 0, 0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

# 파일 확인해보니 도대체 왜 이 부분으로 인해 아프리카 코끼리라는 결과가 도출되는지는 알 수 없다...
write_heatmap(heatmap, "elephant_heatmap.png")

# 원본 이미지를 방금 얻은 열지도와 겹쳐 놓은 이미지 생성해보기
install.packages("magick")
install.packages("viridis")
library(magick)
library(viridis)

image <- image_read(img_path)
info <- image_info(image)
geometry <- sprintf("%dx%d!", info$width, info$height)

# 열 지도 이미지의 혼합/투명 버전 만들기
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "elephant_overlay.png", width = 14, height = 14, bg = NA, col = pal_col)

# 이미지 포개고 출력
image_read("elephant_overlay.png") %>%
  image_resize(geometry, filter = "quadratic") %>%
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()

# 열 지도를 입혀서 본 이미지는 코끼리 코와 귀 부분에 표시가 돼 있어 알아볼 수 있다!
# 코와 귀를 가지고 아프리카 코끼리임을 알아보는 것 같다...

# 5.5 요약
# 합성망은 시각적 분류 문제 공략에 좋다
# 합성망은 시각적 세계를 모듈 패턴과 계층 구조 학습으로 배운다
# 합성망이 배우는 표현은 검사하기 쉽다
# 시각적 데이터 보강을 사용해 과적합 문제를 줄이자
# 사전 훈련 합성망을 사용해 특징 추출 및 미세 조정을 수행할 수 있어야 한다
# 필터와 열 지도를 시각화해 보는 것은 재미 있다