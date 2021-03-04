"""
2021.02.19
Andrew Ng - Deep Learning
Chater 2. Improving Deep Neural Networks
Week 3. Hyperparameter tuning, Batch Normalization and Programming Frameworks

"""

"""
1. Tuning process
하이퍼 파라미터 중
1st - alpha
2nd - beta, hidden units, mini-batch size
3rd - number 0f layers, learning rate decay

생각하지 않아도 되는 것은(adam을 이용할 때)
beta1 = 0.9, beta2 = 0.999, e = 10^-8 

try random values :
딥러닝의 초기단계에서는 어떤 파라미터가 중요한지 아직 모르기 때문에
지점을 임의로 지정하여 시도해 본다.

coarse to fine :
전체 차원에서, 어떤 파라미터들이 좋은지 임의로 지정하여 시도해보다가
가장 좋은 값이 나올 것으로 예상되는 특정 영역을 확대하여
더 높은 밀도로 다시 최적의 파라미터들을 찾는 방법이다.





2. Using an appropriate scale to pick hyperparameters
알파의 최적값을 찾아보자.
alpha = 0.0001 ... 1 이라면 전체의 범위에서 균일하게 임의로 찾아보는 것보단
log scale에서 찾는 것이 더 합리적이다.

0.0001    0.001     0.01     0.1     1로 구간을 나누고
r = -4 * np.random.rand()
alpha = 10^r 로 지정하여 샘플링하면 된다.


기하급수적 가중평균값을 구하는 베타의 최적값을 찾아보자.
beta = 0.9 ... 0.999
0.9는 10개의 값에서 평균을 구하는 것과 같고
0.999는 1000개의 값에서 평균을 구하는 것과 같다.

1 - beta = 0.1 ... 0.001
0.1      0.01      0.001로 나누고
r을 -3에서 -1 사이의 값이 되게 샘플링하면
1 - beta = 10^r
beta = 1 - 10^r 가 된다.

베타 값을 찾을 때 선형 스케일로 샘플링하는 것이 좋지 않은 이유는
베타가 1과 근접하게 될 수록 결과값에 대한 민감도가 커지기 때문이다.

beta = 0.9000   ->   0.9005 로 변했다면 결과값에 그리 큰 영향을 주지 않지만
beta = 0.999    ->   0.9995 로 변했다면 결과값에 큰 영향을 끼친다.





3. Hyperparameters tuning in practice: Pandas vs. Caviar
 1) babysitting one model _ panda approach
   아주 큰 데이터, 적은 산출 자원일 때 적용
   적은 산출 자원 ? CPU, GPU가 작아서 아주 작은 수의 모델만 트레이닝할 수 있는 경우
   day 0일 때, 파라미터를 임의로 초기화시키고
   day1, 2 ... 경과를 보고 판단하여 하이퍼파라미터를 조정한다.
   ex) learning rate를 줄였다 키웠다, 모멘텀 항 조금 더하기
   
 2) training many models in parallel _ caviar approach
   여러 모델을 각기 다른 하이퍼파라미터로 학습시킨다.





4. Normalizing activations in a network
로지스틱 회귀에선, 입력값의 특성을 일반화시키는 것이 러닝 속도를 높여준다.
평균값 구하기 -> (트레이닝 세트 - 평균값) -> 편차 계산
-> 편차대로 데이터 정규화 시키기
이것은 학습모형을 타원형에서 좀더 원형으로 바꿔줄수 있다.

그렇다면 더 깊은 모델은 어떨까 ?
w3, b3을 트레이닝하고 싶을 때 평균값과 a2의 분산을 정규화하면
좀 더 효율적이지 않을까 ?
어떤 숨겨진 층에 대해, 그 전층의 입력값을 정규화 -> 배치 정규화의 역할





5. Fitting Batch Norm into a neural network
배치 정규화는 복잡한 수식으로 이루어져 있지만
딥러닝 프레임워크에서 보통 한줄의 코드로 실행 가능하다.

실제로, 배치 정규화는 보통 트레이닝 세트의 미니 배치와 같이 적용된다.
배치 기울기 강화처럼 전체 트레이닝 세트에서 한번에 트레이닝되는 것 X

  1) 첫번째 미니 배치에서 w1과 b1을 가지고 z1을 계산한다.
  2) z1의 평균값과 분산을 구한다.
  3) (평균값 - 배치 정규화값) / 표준편차  -->  베타1, 감마1로 rescale 
  4) 3단계에서 나온 z1 tilde로 activation함수 적용하여 a1을 구한다.
  5) 두번째 비니배치의 w2와 b2를 가지고 다시 1단계부터 반복한다.

파라미터 : w1, b1, beta1, gamma1 이지만
배치 정규화 이후 b1은 어떤 값을 가지더라도 상수이기때문에
더해진 이후 평균값이 빼지므로 그 효과가 캔슬된다.
따라서 배치 정규화에서 b는 삭제할 수 있다.

z[l] 의 차원은 (n[l], 1)
b[l] = (n[l], 1)
beta[l] = (n[l], 1)
gamma[l] = (n[l], 1)






6. Why does Batch Norm work?
그 이유는 두가지이다.
  1) 모든 특성을 정규화함으로써, 입력 특성 x가 비슷한 범위의 값을 주고,
     이로 인해 러닝의 속도가 높아진다.
  2) weight를 만들기 때문에 속도가 높아진다.

cat classifier를 만들고 검은색 고양이 사진으로만 학습시켰다고 해보자.
이것을 색깔이 있는 고양이 사진에 적용하려면
똑같은 잘 작동하는 함수라고 해도 결과가 좋지 않을 수 있다.
때문에 이런 데이터의 분포가 변하는 것을 covariate shitf라고 한다.
x의 분포도가 변경되면 알고리즘을 다시 트레이닝 시켜야할 수 있다는 것이다.

covariate shift 문제가 신경망에 적용될 때 
배치 정규화는 숨겨진 유닛의 분포가 변동하는 양을 줄여주게 된다.
z2[2], z[2]1의 값이 변하더라도
그것들의 평균값과 분산을 똑같을 것이기 때문이다.(그대로 0과 1의 값을 유지)

이전의 층에서 파라미터가 업데이트 되면서 다음 층에 줄 수 있는 영향을 줄여준다.
때문에 더 안정화되고 다음의 층들이 그 자리를 지킬 수 있게 해준다.
그렇게해서 각각의 층이 스스로 러닝을 진행할 수 있도록 좀 더 독립적으로 변한다.
이것이 전체 네트워크의 속도를 올려준다.





7. Softmax Regression
Softmax Regression - 로지스틱 회귀의 일반화된 버전
고양이(class1) 강아지(class2) 병아리(class3)를 인식한다고 해보자.
그 외(class0)
4개의 클래스가 있는 경우 (0 ~ (class -1))이고
n[l] = 4일 것이다.
입력과 출력의 shape이 같다.

softmax <-> hardmax
hardmax는 가장 큰 값을 1, 나머지는 0으로 출력하는 함수이다.







"""