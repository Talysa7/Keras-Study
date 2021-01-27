# 3.4 영화 감상평 분류: 이항 분류 예제
# IMDB 데이터셋 사용

# 정리정돈
path <- "C:/study/lecture/keras-study"
setwd(path)
rm(list=ls())

# 3.4.1 데이터셋 적재
# %<-% 다중 할당 연산자 사용
library(keras)
imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

# train_data와 test_data는 감상평을 담은 단어 인덱스 리스트
# train_labels와 test_labels는 0(부정)과 1(긍정)로 구성된 리스트
# 구조 확인
str(train_data[[1]])
str(train_labels[[1]])

# num_words = 10000: 훈련 데이터에서 출현 빈도 기준으로 상위 1만 개 단어만 유지한다는 의미
# 희박하게 나오는 단어를 제외해 처리 용량에 맞춰 벡터 데이터 구성 가능
# sapply는 주어진 벡터나 배열과 같은 길이의 리스트를 반환
# 9999개로 1만 개를 넘지 않음
?sapply
max(sapply(train_data, max))

# 감상평 중 하나를 영어 단어로 빠르게 복호화하는 예제
# word_index는 단어를 정수 인덱스에 사상하는 이름이 부여된 리스트
word_index <- dataset_imdb_word_index()
# 정수 인덱스를 단어에 사상
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
# 감상평 복호화
decode_review <- sapply(train_data[[1]], function(index) {
    # 0, 1, 2는 채우기, 시퀀스 시작, 알려지지 않음을 위해 예약된 인덱스이므로 3만큼 인덱스 오프셋
    word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
    if (!is.null(word)) word else "?"
})
# null인 경우 ?로 바뀌어 있음
str(decode_review)

# 3.4.2 데이터 준비
# 정수 리스트를 신경망에 공급(feed)할 수 없으므로 텐서로 변환해야 한다고...
# one-hot encoding으로 처리하는 방법 사용

# 정수 시퀀스를 이진 행렬로 부호화하기
vectorize_sequences <- function(sequences, dimension = 10000) {
    # (길이, 차원) 모양이고 모두가 0인 행렬 생성
    results <- matrix(0, nrow = length(sequences), ncol = dimension)
    # 해당 행렬의 [i]의 특정 인덱스를 1로 설정
    for (i in 1:length(sequences))
        results[i, sequences[[i]]] <- 1
    results
}

# 위 함수로 훈련용, 테스트용 데이터 처리
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# 만들어진 표본 구조 확인하기
# [1:10000] 1 1 0 1 1 1 1 1 1 0 ...
str(x_train[1,])

# 원래는 어땠더라...
str(train_data[1])

# 레이블을 정수에서 숫자형으로 변환하기
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

# 확인해보기
# 자료형이 int에서 num으로 바뀌었다
str(train_labels[1])
str(y_train[1])

