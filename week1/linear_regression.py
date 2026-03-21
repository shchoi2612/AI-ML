import tensorflow as tf
import numpy as np

# y = 2x - 1 데이터셋 준비
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 1개의 뉴런을 가지는 간단한 신경망 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 최적화 함수(SGD)와 손실 함수(MSE) 설정
model.compile(optimizer='sgd', loss='mean_squared_error')

# 모델 학습 (500번 에포크)
print("모델 학습을 시작합니다...")
model.fit(xs, ys, epochs=500, verbose=0)
print("학습이 완료되었습니다!")

# 새로운 값(x=10)에 대한 예측
# y = 2(10) - 1 = 19 이므로 19에 가까운 값이 나와야 합니다.
prediction = model.predict(np.array([10.0]))
print(f"x=10 일 때의 예측값: {prediction[0][0]:.4f}")
