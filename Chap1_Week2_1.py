"""
2021.02.02
Andrew Ng - Deep Learning
Chater 2. 신경망과 로지스틱 회귀
Week 2

"""


"""
### 이진 분류(Binary Classification)
ex) 고양이 사진 입력 받았을 때, 고양이로 알아보았으면 1 아니라면 0(레이블 y)

1. 그 전에, 사진 파일은 어떻게 컴퓨터에 저장될까 ?
- Red, Green, Blue 채널과 일치하는 세개의 행렬을 따로따로 저장한다.
입력된 사진이 64*64 픽셀이라면, 세개의 64*64행렬들이 빨간, 초록, 파랑 픽셀의 강도 값을 알려준다.
Q1) 그러니까 컴퓨터에 사진 파일이 저장되는 방법은 
이 픽셀 강도 값들을 특성 벡터로 바꾸려면, 모든 픽셀값을 특성 벡터 x의 한 열로 나열해야 한다.
주어진 사진에 대한 빨강, 초록, 파랑의 픽셀값 전부를 특성 벡터 x에 나열하면
x는 64*64*3의 차원을 갖게 된다.
이진 분류의 목표는 입력된 사진을 나타내는 특성 벡터 x를 가지고
그에 대한 레이블 y가 1 아니면 0, 즉 고양이 사진인지 아닌지를 예측할수 있도록 학습하는 것이다.

2. 자주 쓸 Notation)
x = n_x차원 상의 특성 벡터
레이블 y = 0 or 1
첫 번째 훈련 샘플 = (x^(1), y^(1))
훈련 샘플의 개수 = m
때때로 무엇을 의미하는지 강조하기 위해 m = m_train
테스트 세트의 개수 = m_test
훈련 샘플들 x를 묶은 행렬 X = X의 열들을 입력된 훈련세트로, X의 행들은 n_x
따라서 행렬 X = n_x * m
레이블 y를 묶은 행렬 Y = [(y^(1)), (y^(2)), ... (y^(m))]
따라서 행렬 Y = 1 * m 이다.


"""



"""
### 로지스틱 회귀(Logistic Regression)
입력될 특성 x가 있다고 하자.
고양이 사진 x가 주어졌을 때 y의 예측값을 출력하는 알고리즘을 원한다.

(w = n_x차원의 벡터, b = 실수)
입력 x와 파라미터 w, b가 주어졌을 때, 어떻게 y의 예측값을 얻을까 ?
y의 예측값은 y가 1일 확률이기 때문에 항상 0과 1 사이어야 한다.
로지스틱 회귀에서는 y의 예측값이 시그모이드 함수를 적용하여 출력된 값이다.
시그모이드 함수 : 수평축이 z일 때, z의 시그모이드는 0부터 1까지 매끈하게 올라가고
                (0, 0.5)를 지난다. 여기서 z는 (w의 전치 * x + b)를 의미한다.
시그모이드(z) = 1 / (1 + e^(-z)) 이지만 z가 아주 큰 수라면
e^(-z)가 0으로 수렴한다.
그러므로 z의 시그모이드는 1 / (1과 0에 가까운 숫자의 합), 1과 가까우므로
z가 아주 크면 z의 시그모이드가 1에 수렴한다.
반면 z가 아주 작은 음수일 경우에는 z의 시그모이드는
1 / (1 + e^(-z))이므로 e^(-z)가 아주 큰 수가 된다.
따라서 z가 아주 작은 음수이면 z의 시그모이드는 0에 수렴한다.

그러므로, 로지스틱 회귀를 구현할 때 y가 1일 확률을 잘 예측하기 위해서는
파라미터 w와 b를 학습해야한다.

"""





"""
### 로지스틱 회귀의 비용 함수(Cost Function)
// 앞으로 x, y, z ... 의 위에 i가 괄호 안에 위첨자로 있으면
ex) x^(i), i번째 훈련 샘플에 대한 데이터임을 뜻하는 것이다.

1. 측정할 수 있는 손실 함수 또는 오차 함수
L(손실 함수)는 출력된 y의 예측값과 참값 y 사이에 오차가 얼마나 큰지 측정한다.
알고리즘이 출력한 y의 예측값과 참 값 y으로
1/2(y^ - y) ** 2 = 손실함수(L)라고 정의할 수 있지만,
로지스틱 회귀에서는 보통 사용하지 않는다.
이는 매개 변수들을 학습하기 위해 풀어야할 최적화 함수가 볼록하지 않기 때문에
여러개의 지역 최적값을 가지고 있어 문제가 되기 때문이다.

최적화 문제가 볼록해지는 또 다른 손실 함수를 정의해보자.
L(y^, y) = -(ylogy^ + (1-y)log(1-y^)) 이다.
이 함수를 쓰는 이유는 첫번째로 y가 1일 경우,
If y = 1 : L(y^, y) = - logy^(L은 그냥 -log)가 된다.
이는 y가 1이기 때문에 두 번째 항 (1-y)가 0이 되기 때문이다.
따라서 logy^의 값이 커지려면 y^ 또한 최대한 커야한다.
하지만 y^은 시그모이드 함수값이기 때문에 1보다 클 수 없다.
그러므로 y = 1일 경우 y^이 1에 수렴하길 원한다는 뜻이다.

두번째로 y가 0일 경우, L의 첫 항이 0이 된다.
그러면 두번째 항이 남아 L(y^, y) = (1-y)log(1-y^))이 된다.
따라서 학습 중에 손실 함수값을 줄이고 싶다면
log(1-y^)이 최대한 커야하므로 y^는 0에 수렴해야하기 때문에
그에 맞춰 매개 변수들을 조정할 것이다.

y가 1일 때 y^가 크고, y가 0일 때 y^가 작은 성질을 가진 함수는 많다.

마지막으로 손실 함수는 훈련 샘플 하나에 관하여 정의되어서
그 하나가 얼마나 잘 예측됐는지 측정해 준다.

비용 함수(J) ?
훈련 세트 전체에 대해 얼마나 잘 추측되었는지 측정해주는 함수
J는 매개 변수 w와 b에 대해, 손실 함수를 각각의 훈련 샘플에 적용한 값의
합들의 평균, 즉 m으로 나눈 값이다.

요점은 손실 함수가 하나의 훈련 샘플에 적용 된다는 것이고
비용 함수는 매개 변수의 비용처럼 작용된다는 것이다.
결과적으로 로지스틱 회귀 모델을 학습하는 것이란,
손실함수를 최소화해주는 매개 변수들 w, b를 찾는 일이다.
흥미롭게도 로지스틱 회귀는 아주 작은 신경망과 같다.

"""





"""
### 경사하강법(Gradient Descent)
w, b를 알아내기 위해서는 비용함수를 가장 작게 만드는 w, b를 찾아야한다.
두 가로축을 나타내는 w, b에서 비용함수 J를 나타내면 보자기처럼 볼록한 모양이다.
J가 볼록하다는 이유가 로지스틱 회귀에 J를 사용하는 가장 큰 이유 중 하나이다.
매개변수에 쓸 좋은 값을 찾기 위해서 w, b를 어떤 값으로 초기화해야 함.
이 초기점에서 시작해 가장 가파른 내리막(가장 빨리 내려올수 있는) 방향으로
한 단계씩 내려가면 언젠가 전역 최적값이나 그 근사치에 도달하게 된다.
w : w − α(dJ(w, b) / dw)
b : b − α(dJ(w, b) / db)
α는 학습률을 뜻하고 (dJ(w, b) / dw)는 미분계수이며 dw라고 한다.
미분계수는 J의 위치에서 함수의 기울기라고 정의할 수 있다.

매개변수가 w만 있는 것으로 가정해보자.
경사 하강법을 큰 w(오른쪽)에서 시작한 경우에, 미분계수는 양수이므로
w에서 미분계수를 빼는 것이 되어서 서서히 왼쪽으로 간다.
왼쪽에서 시작한 경우에는 이와 반대로 미분계수가 음수가 되어
α에 음수를 곱한 값을 빼게 되므로 서서히 증가하여 오른쪽으로 간다.
따라서 어느쪽에서 시작하든 매개변수는 전역 최솟값까지 도달하게 된다.

미적분의 ∂ Notation )
∂J(w, b) / ∂dw에서 ∂는 소문자 d를 다른 폰트로 쓴 것 뿐인데,
이 식은 단수히 J(w, b)의 미분계수를 의미한다.
즉 J가 w 방향으로 얼마나 기울었는지를 나타낸다. 
J가 두개 이상의 변수를 가진 함수일 때 d 대신 사용한다.


"""





"""
### 미분(Derivatives)
1. 도함수란 ?
f(a) = 3a,
a = 2      ->  f(a) = 6
a = 2.001  ->  f(a) = 6.003

a를 오른쪽으로 0.001 밀었을 때 f(a)는 0.003 증가한다.
f가 올라간 정도는 a를 오른쪽으로 민 정도보다 3배 많다.
그렇다면, a = 2에서 함수 f(a)의 기울기, 도함수는 3이다.
도함수는 기울기라고 생각하면 된다.
2. 기울기란 ?
함수가 이동하면서 생긴 작은 삼각형의 높이를 밑변으로 나눈 값.
즉, 0.003 / 0.001 이다.
기울기, 미분계수가 3이라는 것은 a를 아주 조금만 움직여도
f(a)는 그것의 3배정도 커진다는 뜻이다.

    1. df(a) / da = 3
    2. (d / da) * f(a)

도함수를 조금 더 수학적으로 정의하려면
a를 오른쪽으로 훨씬 작은 값만큼 밀었을 경우로 생각해야 한다.
무한대로 작고 작은 양만큼이다.

"""





"""
### 더 많은 미분 예제들(More Derivative Examples)
f(a) = a^2,
a = 2       ->   f(a) = 4
a = 2.001   ->   f(a) = 약 4.004

a = 2일 때 f(a)의 기울기, 미분계수는 4이다.
이 함수(a^2)에서는 a의 값이 다르면 기울기도 다르다.

a = 5       ->   f(a) = 25
a = 5.001   ->   f(a) = 약 25.010
a = 5일 때 f(a)는 10배나 증가하였다.
이 때의 기울기와 미분계수는 10이다.

함수의 곡선 위의 다른 위치마다
높이/밑변의 비율이 다르기 때문에 기울기도 달라진다.
실제로 a^2의 미분계수는 2a라고 정의되어 있다.

+ ) a^3의 미분계수는 3a^2이다.

f(a) = log(a),
log(a)의 도함수는 1 / a인데
a = 2       ->    f(a) = 약 0.69315
a = 2.001   ->    f(a) = 약 0.69365
이는 약 0.0005 증가한 것이다.
a = 2일 때 f(a)의 도함수는 1 / 2이 되는데
0.001의 절반은 0.0005이므로 맞아 떨어진다.


"""




"""
### 계산 그래프(Computation Graph)
1. 신경망의 계산
- 정방향 패스, 정방향 전파
- 역방향 패스, 역방향 전파 로 나뉘어진다.
전방향 패스는 신경망의 출력값을 계산하고
이는 역방향 패스로 이어져 경사나 도함수를 계산한다.
계산 그래프를 보면 왜 이렇게 나누는 지를 알 수 있다.

2. 계산 그래프
변수 a, b, c를 가진 함수 J를 계산한다고 해보자.

J(a, b, c) = 3(a+(b*c))

이 함수를 계산하는 데에는 서로 다른 세 단계의 과정이 필요하다.
먼저 b*c를 계산한 뒤
u라는 변수에 저장하여 u = b*c 라고 하고,
그 값과 a를 더한 결과값을 u에 저장하여 v = a+u라고 해보자.
마지막으로 J = 3v로 표현할 수 있다.
위의 세 단계를 다음과 같이 계산 그래프로 나타낼 수 있다.

a                  -> [v 
b   ->  [u         ->    = a+u]     -> [J = 3v]
c   ->     = bc]

계산 그래프는 J와 같은 특정한 출력값 변수를 최적화하고 싶을 때 유용하다.

"""




"""
### 계산 그래프로 미분하기(Derivatives With Computation Graphs)
계산그래프로 v에 대한 함수 J의 도함수를 구해보자.
 ==> v의 값이 아주 조금 바뀌었을 때 J의 값이 어떻게 바뀌느냐 ?

a                  -> [v 
b   ->  [u         ->    = a+u]     -> [J = 3v]
c   ->     = bc]

J = 3v
v = 11    ->   v = 11.001
J = 33    ->   J = 33.003
v에 대한 J의 도함수는 3이다.

a의 값이 바뀌면 J의 값은 어떻게 될까 ?
a = 5     ->   a = 5.001
v = 11    ->   v = 11.001
J = 33    ->   J = 33.003
J의 도함수는 3이다.
a 값의 변화는 v의 값을 변하게하고
v의 변화는 J를 증가시킨다.(미적분의 연쇄법칙)

마지막 출력값 변수의 v에 대한 도함수를 얻는 것이 
그래프에서 한 단계 뒤로 이동하여
a에 대한 J의 도함수를 계산하는데 도움을 줄 수 있다는 것이다.
이것이 역전파의 한 단계이다.

이 예시에서는 최종 출력값이 J이지만 소프트웨어에서는 ?
dvar을 사용하여 구하고자 하는 최종 출력값 변수,
여러 중간 값에 대해 계산한 도함수를 표현한다.

 
"""



"""
### 로지스틱 회귀의 경사하강법(Logistic Regression Gradient Descent)
계산그래프를 로지스틱 회귀의 경사하강법에 사용해보자.

손실함수 L의 역방향으로 가서 a에 대한 손실함수의 도함수를 계산해보자.
코드에서는 이 변수를 da라고 나타낼 수 있다. 계산해보면,
da = -(y/a) + ((1-y) / (1-a)) 이다.
a에 대한 최종 출력값의 도함수인 da를 계산했으니
한번 더 역방향으로 가서 z에 대한 손실함수의 도함수인 dz를 구할 수 있다.
dz = a - y 이다.
마지막으로 w와 b의 도함수는
dw1 = x1*dz
dw2 = x2*dz
db = dz 이다. 

"""



