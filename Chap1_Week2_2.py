"""
2021.02.07
Andrew Ng - Deep Learning
Chater 3. 파이썬과 벡터화
Week 2

"""


### 1. Vectorization
# 딥러닝은 큰 데이터세트에서 유리하다.

## 벡터화란 ?
import numpy as np
a = np.array([1, 2, 3, 4])

# Vectorized Version
import time
# 백만 디멘션 만들기 _ np.random.rand는 0 ~ 1 사이의 균일 분포 값을 리턴한다.
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
# np.dot => 두 벡터의 내적(Dot product)를 계산한다.
c = np.dot(a, b)
toc = time.time()

print(c)
print('Vectorized version : ' + str(1000*(toc-tic)) + 'ms')


# Non - Vectorized Version
c = 0
tic = time.time()
for i in range(1000000) :
    # 일일이 계산.
    c += a[i]*b[i]
toc = time.time()

print(c)
print('For loop : ', str(1000*(toc-tic)) + 'ms')


## Vectorized Version이 500배 가까이 빠르다.
##  -> 큰 데이터 세트를 다루는 딥러닝에서, 이 차이는 굉장이 크다.
##  -> 벡터화를 하는 이유, 코드의 속도를 현저하게 높일 수 있다.

# Q. True or false. Vectorization cannot be done without a GPU.
#   -> False
#   -> CPU에서도 가능





### 2. More Vectorization Examples
# 회귀에서 혹은 새로운 네트워크를 프로그래밍할 때, 가능한 한 for-loop을 피해야 한다.

# 어떤 벡터의 모든 요소(n개)에 지수를 적용하고 싶다면
# Non-Vectorized
import math
v = np.random.rand(100)
n = 100
u = np.zeros((n, 1))
for i in range(n) :
    u[i] = math.exp(v[i])

# Vectorized
import numpy as np
u = np.exp(v)


# 그 외 ...
np.log(v)
np.abs(v)
np.maximum(v, 0)  # -> v의 모든 요소에 대한 최대값을 0으로



### 3. Vectorizing Logistic Regression
## 전방향전파
# m개의 트레이닝 세트 중
# 첫 번째 세트
# z^(2) = w^T * x^(2) + b
# a^(2) = σ(z^(2))
# 두 번째 세트
# z^(2) = w^T * x^(2) + b
# a^(2) = σ(z^(2))
# 세 번째 세트
# z^(3) = w^T * x^(3) + b
# a^(3) = σ(z^(3))

# X = [x^(1), x^(2), x^(3), ... x^(m)]    ->   (nx, m) 매트릭스
# Z = w^T * X + [b, b, b, .. b]
#   = [z^(1), z^(2), ... z^(m)]
# 따라서 Z = np.dot(w*T, X) + b 이고
# A = [a^(1), a^(2), ... a^(m)] = σ(Z) 이다.

# Q. What are the dimensions of matrix X in this video?
# (nx, m)





### 4. Vectorizing Logistic Regression's Gradient Output
# dz^(1) = a^(1) - y^(1)
# dz^(2) = a^(2) - y^(2)
# dz = [dz^(1), dz^(2), ... dz^(m)]     ->    m차원의 벡터

# A = [a^(1), ... a^(m)]
# Y = [y^(1), ... y^(m)]

# dz = A - Y = [a^(1) - y^(1), ... a^(m) - y^(m)]

# dw = 0
# dw += x^(1) * dz^(1)
# dw += x^(1) * dz^(2)
# ... dw /= m

# db = 0
# db += dz^(1)
# db += dz^(2)
# ... db /= m

# dw = 1/m * dZ^(T)
# db = 1/m * np.sum(dz)


# Q, How do you compute the derivative of b
#    in one line of code in Python numpy?
# 1 / m*(np.sum(dz))



print('---------------------------------------------------------')



### 5. Broadcasting in Python
## Broadcasting example
# Calories from Carbs, Proteins, Fats in 100g of different foods :
#              Apple   Beef   Eggs   Potatoes
#  Carb    [   56.0    0.0    4.4      68.0  ]
#  Protein |   1.2    104.0   52.0      8.0  |
#  Fat     [   1.8    135.0   99.0      0.9  ]

# 칼로리의 퍼센트를 각각 음식 별로 탄, 단, 지로 나누어 계산해 보자.
# 단, for - loop 없이 !

import numpy as np
A = np.array([[56.0, 0.0, 4.4, 68.0],
             [1.2, 104.0, 52.0, 8.0],
             [1.8, 135.0, 99.0, 0.9]])

print(A)
# 세로로 모두 더하기
cal = A.sum(axis=0)
print(cal)

percentage = 100 * A / cal.reshape(1, 4)   # -> cal을 (3, 4)로 행 복사하여 계산
print(percentage)


# Broadcasting example
# [1 2 3 4] + 100 = [101 102 103 104]
# -> 자동으로 100을 [100 100 100 100] 변환하여 더한다.

# [[1 2 3] [4 5 6]] + [100 200 300] 에서는
# [[100 200 300] [100 200 300]] 으로 변환하여 더한다.
# = [[101 202 303] [104 205 306]]

# [[1 2 3] [4 5 6]] + [[100] [200]] 에서는
# [[100 100 100] [200 200 200]] 으로 변환하여
# [[101 102 103] [204 205 206]] 이 된다.

# General Principle of Broadcasting
# (m, n)    + - * /    (1, n)   ->   (m, n)


# Q. Which of the following numpy line of code
#    would sum the values in a matrix A vertically?
# A.sum(axis = 0)



print('---------------------------------------------------------')



### 6. A note on python/numpy vectors
import numpy as np
# 임의의 5개 가우시안 변수 만들기
a = np.random.randn(5)
print(a)

# a의 모양은 (5,)  ->  rank 1 array = 행, 열 벡터가 아님.
# rank 1 array는 row vector나 column vector과 같이 균일하게 행동 X
# 직관적이지 않다.
print(a.shape)
# a와 a.transpose는 동일해 보임.
print(a.T)
# 이 둘을 dot해보면 행렬이 나오지 않을까 ? but, No.
print(np.dot(a, a.T))

# 따라서 애매모호한 (5,), (n,), rank 1 array를 사용하지 말고
# 직접 세로벡터가 되게끔 지정해주자.
a = np.random.randn(5, 1)
print(a)
# transpose하면 제대로 (1, 5)가 된다.
print(a.T)
# dot해도 제대로 행렬이 출력된다.
print(np.dot(a, a.T))

# +) reshape 함수를 부르는데 주저하지 말자.

# Q. What kind of array has dimensions in this format: (10, ) ?
# A rank 1 array

# Q. True or False: Minimizing the loss corresponds with maximizing logp(y|x).
# True




a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
print(c)


a = np.random.randn(12288, 150) # a.shape = (12288, 150)
b = np.random.randn(150, 45) # b.shape = (150, 45)
c = np.dot(a,b)
print(c.shape)

# a = np.random.randn(4, 3) # a.shape = (4, 3)
# b = np.random.randn(3, 2) # b.shape = (3, 2)
# c = a*b
# print(c)


a = np.random.randn(2, 3) # a.shape = (2, 3)
b = np.random.randn(2, 1) # b.shape = (2, 1)
c = a + b
print(c)

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
print(c)


