# 과제 1: 환경 설치 및 프로그램 실행

**작성자:** 최상현 (202312162, 물리학과)

## 실행 환경

- Python 3.12.3
- TensorFlow 2.20.0
- NumPy 2.3.5
- Matplotlib 3.10.7
- ReportLab 4.4.10

## 실행 결과

### 00_hello_world.py — 환경 확인

```
========================================
Hello World! Environment Check
========================================
Python Version: 3.12.3
----------------------------------------
✅ numpy: Installed (Version 2.3.5)
✅ matplotlib: Installed (Version 3.10.7)
✅ tensorflow: Installed (Version 2.20.0)
✅ reportlab: Installed (Version 4.4.10)
========================================
If you see all checkmarks, your environment is ready!
```

### 01_hello_nn.py — 단순 신경망 (y = 2x - 1 학습)

```
TensorFlow Version: 2.20.0

Training Data:
X: [-1.  0.  1.  2.  3.  4.]
y (clean): [-3. -1.  1.  3.  5.  7.]
y (noisy): [-2.50328585 -1.1382643   1.64768854  4.52302986  4.76584663  6.76586304]

Starting training...
Training finished!

Prediction for x=10.0: 18.5816
Expected value: 19.0

Learned Parameters:
Weight (w): 1.9099 (Expected: 2.0)
Bias (b): -0.5174 (Expected: -1.0)
Formula: y = 1.9099x + -0.5174
```

생성된 그래프:
- `outputs/training_loss.png` — 학습 손실(Loss) 변화 그래프
- `outputs/model_fit.png` — 신경망 피팅 결과

### 02_polynomial_fitting.py — 수치 해석 vs 신경망 비교

```
==================================================
Numerical Methods vs Neural Networks
==================================================

[Method 1] NumPy Polyfit (Least Squares)
Result: y = 1.9124x + -0.5251
Prediction for x=10.0: 18.5987

[Method 2] SciPy Curve Fit (Optimization)
Result: y = 1.9124x + -0.5251
Prediction for x=10.0: 18.5987

Summary:
Neural Network (Previous): Iterative learning (Gradient Descent)
NumPy Polyfit: Analytical solution (Linear Algebra)
SciPy Curve Fit: Numerical optimization (Levenberg-Marquardt)
```

생성된 그래프:
- `outputs/02_numerical_fitting.png` — 다항식 피팅 결과 비교
