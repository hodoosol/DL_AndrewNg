"""
2021.02.17
Andrew Ng - Deep Learning
Chater 2. Improving Deep Neural Networks
Week 2. Optimization algorithms

"""


"""
1. Mini-batch gradient descent
벡터화는 모든 m 샘플들에 대해서 효율적으로 계산할 수 있게 해준다.
X = [x1, x2, x3 ... xm]   ->   (nx, m)
Y = [y1, y2, y3 ... ym]   ->   (1, m)

만약 m = 5000000 라면 ?
 x를 1000개 단위로 나누고 x{1}, x{2} ... x{5000}으로 묶어준다.
 이것을 mini-batch라고 한다. y에도 동일하게 적용해준다.
 x의 mini-batch인 x{t}들은 각각 (nx, 1000)의 모양을 가진다.
 
for t = 1 ... 5000
 : 1번 당 1000개의 예시를 한번에 처리한다.
 Z[1] = W[1]X{t} + b[1], A[1] = g[1](Z[1])
 그 다음엔 비용함수 J를 계산한다.
 
아주 큰 트레이닝 세트가 있는 경우, 
mini-batch가 그냥 batch 기울기 강하보다 훨씬 더 빨리 운영된다.





2. Understanding mini-batch gradient descent
mini-batch는 batch 기울기 강하보다 노이즈가 있다.
그 이유는 X{1}, Y{1}가 조금 더 쉬운 mini-batch여서 비용이 조금 더 낮거나,
우연으로 X{2}, Y{2}가 어려운 mini-batch여서 비용이 조금 더 높기때문일 수 있다.

우리가 골라야하는 파라미터 중 하나는 미니배치의 크기이다.
만약 미니배치의 사이즈가 m과 같다면 이것은 전체 트레이닝 세트와 동일하기 때문에
단순한 배치 기울기강하를 준다.

만약 미니배치 사이즈가 1이라면 stochastic 기울기 강하라는 알고리즘을 준다.
각각의 예시는 모두 미니배치이다. 하나의 트레이닝 샘플씩 강하하게 된다.
stochastic은 절대 수렴하지 않기 때문에 항상 최소값 범위 사이에서 움직인다.

우리가 사용할 미니배치는 이 둘 사이의 어떤 것일 것이다.
그렇다면 최적의 미니배치 값은 어떻게 찾을까 ?
 1. 작은 트레이닝 세트일 경우, 그냥 배치 기울기 강하를 이용해라. (2000 이하)
 2. 조금 큰 트레이닝 세트일 경우 64, 128, 245, 512(2의 지수값)이 좋다.
 3. 모든 X{t}, Y{t}가 CPU/GPU 메모리에 들어가게 하도록 하는것이 좋다.
 




3. Exponentially weighted averages
지수적 가중평균
Vt = (Beta * Vt-1) + (1 - Beta) * theta
Vt의 값은 보통 이전의 (1 / 1 - beta)일 동안의 평균 기온과 비슷하다.
ex ) beta = 0.9 라면 이전 10일 동안의 평균 기온
     beta = 0.98 라면 이전 50일 동안의 평균 기온  ->  완만한 그래프
     beta = 0.5 라면 이전 2일 동안의 평균 기온  -> 들쭉날쭉한 그래프(noisy)





4. Understanding exponentially weighted averages
vt = (Beta * Vt-1) + (1 - Beta) * theta
Beta = 0.9
v100 = 0.9 * v99 + 0.1(theta100)
v99 = 0.9 * v98 + 0.1(theta99)
v98 = 0.9 * v97 + 0.1(theta98)

v100 = 0.1(theta100) + 0.9(0.1(theta99) + 0.9(0.1(theta98 + 0.9 * v97 + )) ....
beta = 0.9일 때, 10일 정도 이후 현재 날짜 가중치의 1/e 보다 작아짐.
beta = 0.98일 때는 50일 이후.


v = 0 (초기화)
Vt = (Beta * Vt-1) + (1 - Beta) * theta1
Vt = (Beta * Vt-1) + (1 - Beta) * theta2 
...

매일 매일 새로운 vt값으로 갱신한다.
따라서 단 하나의 값만 메모리에 보관하면 되는 것이 지수 가중 평균의 장점이다.
마지막 구한 값에 공식을 써서 덮어쓰면 된다. 메모리 효율이 좋다.





5. Bias correction in exponentially weighted averages
만약 초기 구간의 바이어스를 신경쓴다면, 
지수 가중 평균의 추정치를 더 정확하게 만드는 방법은
vt 대신에 vt / (1-beta^t) 을 사용하는 것이다.





6. Gradient descent with momentum
경사 하강법을 적용할 때,
가로축의 경우에는 빠른 러닝을 원할 것이고
세로축의 경우에는 변동이 싫기때문에 러닝 속도가 조금 늦길 바랄 것이다.
모멘텀을 이용하여 기울기 강하를 빠르게 실행할 수 있다.





7. RMSprop
모멘텀과 비슷하게, 기울기강하, 미니배치 기울기 강하에서
변동을 무디게하는 효과가 있다.
때문에 조금 더 큰 러닝 속도 알파 값을 사용할 수 있게 되고
알고리즘의 러닝 속도를 높여준다.





8. Adam optimization algorithm
adam = 모멘텀 + rmsprop

1. 초기화. Vdw, Sdw, Vdb, Sdb = 0
2. 반복 루프 t회에 걸쳐 미분 계산
3. 모멘텀 지수 가중 평균 Vdw = B1 * Vdb + (1 - B1)db
4. rmsprop Sdb = B2 * Sdb + (1 - B2)db
5. 편향 보정
6. W, b 업데이트

하이퍼 파라미터는 ?
alpha
beta1, beta2
E





9. Learning rate decay
학습 알고리즘을 가속화시키는데 도움이 되는 것은
시간이 가면서 러닝 레이트를 천천히 줄여나가는 것이다.

미니배치 경사하강은 최소값으로 가는 경향은 있지만 수렴하지는 않는다.
이는 알파값을 고정했기 때문이며, 미니배치 고유의 노이즈도 있기 때문이다.
만약 학습속도인 알파를 천천히 줄여나간다면 
초기에는 빠르게 학습하다 최소값 부근의 아주 작은 값 사이에 머물 것이다.

1 epoch = 1 pass 트레이닝 세트의 학습 루프를 1번 돈 것.
alpha0 = 알파의 초기값
alpha = (1 / (1 + decay-rate * epoch-num)) * alpha0

epoch 1 : alpha = 0.1 
epoch 2 : alpha = 0.67 
epoch 3 : alpha = 0.05 
epoch 4 : alpha = 0.04
...
처럼 줄어든다.


지수 감쇠법
alpha = (0.95^epoch-num) * alpha0





10. The problem of local optima
초기 딥러닝에서는 최적화 알고리즘이 안좋은 국소 최적값에 걸리는 것을 걱정했다.
그러가 기울기가 0인 지점이 항상 국소 최적값인 것은 아니다.
20000개의 매개변수가 있을 때, 20000 차원의 벡터에서 정의되는 J 함수가 있고
이 함수를 통해 국소 최적값을 보기보다는 안장점을 볼 확률이 더 높다.

국소 최적값이 큰 문제가 아니라면 무엇이 문제일까 ?
plateau - 함수 기울기의 값이 0에 근접한 긴 범위
이 지점에 있다고하면 기울기 강하가 표면을 따라 밑으로 이동하게 되고
정체구간에 도달하는데에는 굉장히 오랜 시간이 걸릴 것이다.
이 구간이 바로 모멘텀, rmsprop, 아담 알고리즘이 도움을 줄 수 있는 부분이다.






"""