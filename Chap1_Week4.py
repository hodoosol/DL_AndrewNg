"""
2021.02.15
Andrew Ng - Deep Learning
Chater 4. 얕은 신경망
Week 3

"""

### 1. Deep L-layer neural network
# 깊은 신경망이란 ?
# 로지스틱 회귀는 보통 얕은 모델이라고 표현한다. -> 1개 층 신경망
# 숨겨진 층이 많을수록 깊은 신경망

# 표기법
# L = 층의 개수 ex) 4
# n[l] = l층에서의 유닛 개수 ex) n[2] = 5 (2번째 층의 유닛은 5개)
# n[0] = 입력층 = nx ex) 3 (x1, x2, x3이 있을 때)
# a[l] = l층에서의 활성함수 = g[l](z[l])
# w[l] = z[l]의 weights
# x = a[0], y^=a[l]




### 2. Forward Propagation in a Deep Network
# x : z[1] = w[1]x + b[1], a[1] = g[1](z[1]) -> 첫 번째 layer
# z[2] = w[2]a[1] + b[2], a[2] = g[2](z[2])  -> 두 번째 layer

# z[l] = w[l]a[l-1] + b[l], a[l] = g[l](z[l])
# 이것을 벡터화하면,
# Z[1] = W[1]A[0] + b[1], A[1] = g[1](Z[1])
# Z[2] = W[2]A[1] + b[2], A[2] = g[2](Z[2])
# y^ = g(Z[4]) = A[4]  ->  이 예측 수치는 모든 트레이닝 샘플들을 가로로 쌓은 것이다.





### 3. Getting your matrix dimensions right
# L= = 5
# z[1] = w[1]x + (b[1] ... 일단 b는 무시)
# w[1]은 n[1] * n[0]의 모양을 가진다.
# 일반화하면, w[l] = n[l] * n[l-1]이다. == dw[l]

# b[1] = n[1] * 1
# 일반화하면, b[l] = n[l] * 1           == db[l]

# z[l] = g[l](a[l])이므로 나머지의 디멘션도 주의깊게 봐야한다.
# 정리하자면,
#  z[1]  =  (w[1]   *   x)   +   b[1]
# (n[1],1) (n[1], n[0]) (n[0],1) (n[1], 1) 의 shape을 가진다.

#  Z[1]  =  W[1]    *    X   +   b[1]
# (n[1],m) (n[1],n[0]) (n[0],m) (n[1],1)

# z[l], a[l] : (n[l], 1)
# Z[l], A[l] : (n[l], m)     == dZ[l], dA[l]
# l = 0일 때는, A[0] = X = (n[0], m)





### 6. Forward and Backward Propagation

## Forward Propagation
# input a[l-1]
# output a[l], cache(z[l])

# z[l] = w[l]*a[l-1] + b[l]
# a[l] = g[l](z[l])

# 벡터화 하면
# Z[l] = W[l]*A[l-1] + b[l]
# A[l] = g[l](Z[l])

## Backward Propagation
# input da[l]
# output da[l-1], dW[l], db[l]

# dz[l] = da[l] * g[l](z[l])
# dw[l] = dz[l] * a[l-1]
# db[l] = dz[l]
# da[l-1] = w[l].T * dz[l]
# dz[l] = w[l+1].T * dz[l+1] * g[l](z[l])

# dz[l] = w[l]*a[l-1] + b[l]
# a[l] = g[l](z[l])

# dZ[l] = dA[l] * g[l](Z[l])
# dW[l] = 1/m(dZ[l] * A[l-1].T)
# db[l] = (1/m)np.sum(dZ[l], axis=1, keepdims=True)
# dA[l-1] = W[l].T * dZ[l]


# Summary
# x -> Relu -> Relu -> Sigmoid -> y^ -> L(y^, y)
#  dw[l], db[1] <- dw[2], db[2] <- dw[3], db[3]

# 전방향에서는 입력데이터 x를 넣어서 초기화 한다.
# da[l] = (-y/a) + (1-y) / (1-a)로 초기화 한다.

# 만약 벡터화된 도입을 구하는 경우 backward recursion을 초기화 한다.
# dA[l] = (-y(1)/a(1)) + (1-y(1)) / (1-a(1)) ... (-y(m)/a(m)) + (1-y(m)) / (1-a(m))
# 이렇게 벡터화된 버전을 도입하면 된다.





### Parameters vs Hyperparameters
# parameters : W[1], b[1], W[2], b[2] ...

# hyperparameters : learning rate alpha
# iterations, hidden layer L, hidden unit n[1], n[2] ...
# choice of activation function
# 이런 하이퍼 파라미터들은 ultimate parameter인 W와 b를 컨트롤한다.
# 그렇기 때문에 하이퍼 파라미터라고 불린다.

# 최적의 하이퍼 파라미터를 찾기 위해 계속해서 반복적으로 여러가지
# 하이퍼 파라미터를 시도해봐야한다.








