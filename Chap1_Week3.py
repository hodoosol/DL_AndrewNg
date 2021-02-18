"""
2021.02.13
Andrew Ng - Deep Learning
Chater 4. 얕은 신경망
Week 3

"""


### 1. Neural Networks Overview
# z = w.T * x + b  ->  a = sigmoid(z)  ->  L(a, y)
# yhat = a



### 2. Neural Network Representation
# Input layer -> Hidden layer -> Output layer -> yhat
# 이전에는 입력값을 x로 표현했지만 a^[0]으로도 가능하다.
# a = activation
# 서로 다른 신경망들이 다음 이어지는 신경망 층들로 전달하는 값을 뜻한다.
# 입력층이 x값을 a^[0]으로 숨겨진층에 전달하고
# 숨겨진층은 a^[1]으로 activation을 생성하고
# 각 노드는 a1^[1], a2^[2] ... an^[n]와 같은 값은 값을 생성한다.
# a^[1]는 (n * 1)차원의 벡터가 된다. n = 숨겨진 층의 노드 개수
# Input layer -> Hidden layer -> Output layer -> yhat 의 경우
# "2 layer NN"이라 불린다. 그 이유는 입력층은 세지 않기 때문이다.
# 입력 특성이 x1, x2, x3으로 3개, 숨겨진 층의 노드가 4개일 때,
# 숨겨진 층은 w(4 * 3), b(4 * 1)의 파라미터를 가지고
# 결과층은 w(1 * 4), b(1, 1)의 파라미터를 가진다.



### 3. Computing a Neural Network's Output
# x1, x2, x3     - 입력
# node1, 2, 3, 4 - 숨겨진 층(layer 1)
# a1^[1] -> 1은 node in layer, [1]은 layer
# node1 : z1^[1] = w1^[1].T * b1^[1], a1^[1] = sigmoid(z1^[1])
# node2 : z2^[1] = w2^[1].T * b2^[1], a2^[1] = sigmoid(z2^[1])
# node3 : z3^[1] = w3^[1].T * b3^[1], a3^[1] = sigmoid(z3^[1])
# node4 : z4^[1] = w4^[1].T * b4^[1], a4^[1] = sigmoid(z4^[1])
# 이것을 신경망에 도입할 때, for loop은 너무나 비효율적. 벡터화해보면
# z^[1] = W^[1] * x + b^[1],      a^[1] = sigmoid(z^[1])
# (4, 1) (4, 3) (3, 1)(4, 1)      (4, 1) (4, 1) 로 이루어져있다.
# Q. 3:30 w가 왜 (4,3)형태일까 ?



### 4. Vectorizing across multiple examples
# 입력 x1, x2, x3  - 출력 yhat
# 공식 !
# z^[1] = W^[1] * x + b^[1]
# a^[1] = sigmoid(z^[1])
# z^[2] = W^[2] * a^[1] + b^[2]
# a^[2] = sigmoid(z^[2])

# x -----> a^[2] = yhat
# x^[1] -----> a^[2](1) = yhat^[1]
# x^[2] -----> a^[2](2) = yhat^[2]
# x^[n] -----> a^[2](n) = yhat^[n]

# unvectorizing
# for i = 1 to m :
#   공식

# vectorizing
# z^[1]과 a^[1]을 Z^[1]과 A^[1]을 얻을 때까지 가로로 쌓는다.
# 이 매트리스들은 트레이닝 샘플들을 가로로 인덱싱하고
# 세로 인덱스는 신경망의 여러 노드를 나타낸다.



### 5. Explanation for Vectorized Implementation
# z^[1](1) = W^[1] * x(1) + b^[1]
# z^[2](2) = W^[1] * x(2) + b^[1]
# z^[2](3) = W^[1] * x(3) + b^[1]
# b는 0으로 만들자.
# w^[1] * x(1) 은 (4, 1) 행렬이다.
# 이것들을 차곡차곡 가로로 쌓으면 z^[1](1)을 세로 벡터로 쌓은 것과 같으며
# 이것은 Z^[1]이다.
# 만약 b가 0이 아니라면 b(i)의 값을 각각 더해주면 된다.
# 이 과정은 모든 샘플 m개에 걸쳐 동시에 벡터화시킬 수 있다.
# 또, Z^[1] = W^[1] * X + b^[1]에서 X는 a^[0]이다.
# 따라서 신경망의 다른 모든 층들이 사실은 비슷한 활동을 하는 것을 알 수 있다.



### 6. Activation functions
# 활성함수 중 시그모이드 함수보다 거의 항상 좋은 성능을 내는 것은
# tanh 함수이다. -> 시그모이드 함수를 평행이동한 것
# 시그모이드함수는 보통 이진 분류에만 사용된다.
# ReLU(rectified linear unit)함수는 a = max(0, z)이다.
# ReLU가 보통 활성함수를 선택하는 기본 사항이다.
# 사용빈도 : ReLU > tanh > sigmoid
# ReLU의 한가지 단점은 z가 음수일때 미분값이 0이라는 것인데,
# 실제로는 잘 작동하지만 보완을 위해 leaky ReLU를 사용하기도 한다.

# sigmoid - 이진분류나, 출력층이 아니라면 사용하지 말자.
# tanh - sigmoid보다 거의 항상 성능이 좋다.
# ReLU - 가장 일반적으로 사용되는 활성함수



### 7. Why do you need non-linear activation functions?
# 왜 우리의 신경망에서 비선형 활성함수가 필요할까 ?
# a[1] = z[1] = w[1] * x + b[1]
# a[2] = z[2] = w[2] * a[1] + b[2]
# 첫 번째 줄의 a[1]을 두 번째 줄의 a[1]에 대입하면
# a[2] = w[2](w[1] * x + b[1]) + b[2] 가 되고
#      = (w[2]*w[1])x + (w[2]b[1] + b[1])
#      = w-prime + b-prime
# 선형에서의 숨겨진 층은 거의 쓸모가 없다.
# 왜냐면 2개의 선형 함수의 구성요소는 그 자체가 선형함수이기 때문이다...
# but, g(z) = z의 경우에서는 선형 함수를 사용한다.



### 8. Derivatives of activation functions
# g(z)가 시그모이드 함수인 경우 기울기는 d/dz*g(z)이다.
# g(z)가 tanh 함수인 경우 기울기는 d/dz*g(z) = 1 - (tanh(z))^2 이다.

# g(z)가 ReLU 함수인 경우 g(z) = max(0, z)이고
# 기울기는 z < 0일 때 0, z > 0일 때 1이다. 0일 때는 undifined.

# g(z)가 Leaky ReLU 함수인 경우 g(z) = max(0.01z, z)이고
# 기울기는 z < 0일 때 0.01, z > 0일 때 1이다. 0일 때는 undifined.



### 9. Gradient descent for Neural Networks
# 우리의 신경망은
# parameters : w[1], b[1], w[2], b[2]
# nx = n[0], n[1], n[2] = 1

# w[1]의 shape은 (n[1], n[0]), b[1]는 (n[1], 1)
# w[2]의 shape은 (n[2], n[1]), b[1]는 (n[2], 1)

# 비용 함수 : J(w[1], b[1], w[2], b[2]) = (1/m) * sigma(i to n)(L(y^, y))
# y^ = a[2]

# 알고리즘의 파라미터를 트레이닝 시키기 위해서는 경사하강법을 진행해야함.



### 10. Random Initialization
# 신경망에서는 weight를 임의로 초기화하는 것이 중요하다.
# 로지스틱 회귀에서는 0으로 초기화해도 괜찮았으나,

# 신경망에서 모든 weight를 0으로 초기화한다면 ?
# 숨겨진 층의 모든 노드가 동일해져서, 동일한 함수를 출력하게 된다.
# 그렇기 때문에 신경망을 아무리 오래 훈련시킨다해도,
# 숨겨진 유닛이 대칭이기때문에 모든 숨겨진 층이 완전히 똑같은 함수를 출력한다.
# 신경망이 크다고 해도 마찬가지 이다.
# 이것에 대한 해결책은 파라미터를 임의로 초기화하는 것이다.

# w[1] = np.random.randn((2, 2)) * 0.01 , b는 0으로 초기화해도 됨.
# 그렇다면 왜 0.01을 곱할까 ?
# w가 매우 크거나 매우 작으면 z도 매우 크거나 매우 작을 것이기 때문에
# 기울기의 값이 아주 작아져서 기울기강하도 매우 느릴 것이다.
# 따라서 초기 파라미터값을 너무 크지 않게 하기위해 0.01을 곱한다.
# 그러나 아주 깊은 신경망의 경우 상수의 값을 달리할 수 있다.



