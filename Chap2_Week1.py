"""
2021.02.17
Andrew Ng - Deep Learning
Chater 2. Improving Deep Neural Networks
Week 1. Setting up your ML Application

"""


"""
1. Train / Dev / Test sets
 학습세트, 개발세트, 테스트세트를 잘 설정하면 좋은 성능의 신경망을 찾는데 좋다.
 그러나 그러기 위해선 매우 반복적인 과정을 실행할 수 밖에 없다.

 전체 데이터세트의 개수가 많아질수록 dec, test 세트의 퍼센트는 낮게 설정하는 추세이다.
 또한,
 training sets - cat pictures from webpages
 dev/test sets - cat pictures from users using your app
 등으로 구분할 수도 있다.




2. Bias / Variance 편향 / 분산
 최근, 편향과 분산의 균형에 대한 토론은 줄어들었다.

 train set error : 1%
 dev set error : 11% 라면 트레이닝세트에 과적합되었다고 볼 수 있다. -> 큰 분산
 
 train set error : 15%
 dev set error : 16% 라면 알고리즘이 트레이닝세트에서도 성능을 못하는 듯 하다.
                     -> underfitting 큰 편향
 
 train set error : 15%
 dev set error : 30% 라면 큰 분산, 큰 편향      
 
 train set error : 0.5%
 dev set error : 1% 라면 적은 분산, 적은 편향 -> happy             
 



3. Basic Recipe for Machine Learning
 우리의 알고리즘이 큰 편향을 띄고 있는가 ?
 Yes or 트레이닝 세트에 잘 맞지 않는다. -> try bigger network or another network
 No -> 큰 편차를 띄고 있는가 ? Yes -> 더 많은 데이터를 수집하거나 regularization
 No -> Done !
 
  
  
  
4. Regularization
 알고리즘이 과적합되었을 때 -> 더 많은 데이터세트를 얻는 것으로 해결할 수 있지만
 그럴수 없는 경우 일반화해야한다.
 
 Logistic regression
  비용함수를 최소화해보자.
  L2 일반화를 L1 일반화보다 자주 사용한다.
  L2의 람다는 일반화 파라미터로 불리며 우리가 값을 조정해주어야하는 하이퍼파라미터이다.
  파이썬에서는 lambd로 사용한다.
  
  Neural network
  비용함수는 손실함수의 합과 같은데,
  일반화는 모든 파라미터 w에 2m 이상의 람다를 더한다.
  프로베느우스 표준형 = 매트릭스 요소의 제곱의 합
  



5. Why regularization reduces overfitting?
 과적합된 모델에 일반화를 적용하면
 L2 norm 또는 Frobenius norm이 줄어드는데
 람다 일반화를 크게 정하면 w의 값을 0에 가깝게 만들고
 수많은 숨겨진 유닛의 영향을 거의 0으로 만든다.
 이 경우 신경망이 작아져 분산이 줄어든다.
 조금 더 심플한 네트워크가 되면 과적합이 줄어들게 된다.
 
 또, 활성함수가 tanh함수 일 때
 람다가 커지면 w가 줄어들고, 이 때문에 z의 값도 작아질 것이다.
 z의 값이 작을 경우 G(z)는 대략적으로 선형이 될 것이다.
 이것은 모든 층이 대략적으로 선형인 셈이 되고,
 모든 층이 선형이라고 하면 전체 네트워크도 선형 네트워크이다.
 그렇기 때문에 복잡한 결정을 피팅하는 것은 불가능하고
 과적합될 확률도 줄어든다.
 
 

6. Dropout Regularization
네트워크의 각층을 살펴보면서 신경망의 노드를 제거하는 확률을 세팅해보자.
각 층의 노드별로 0.5의 확률로 노드를 제거할 것인지 유지할 것인지 정해보면
훨씬 더 작은, 감소된 네트워크가 남을 것이다.

Inverted dropout
 ex) a3 /= keep.prob
 scaling 문제가 적어지기 때문에 test time을 쉽게 만들어준다.
 



7. Understanding Dropout
dropout에서는 무작위로 네트워크의 유닛을 해체한다.
때문에 한 유닛은 어떤 한가지의 피처에 의존하면 안된다.
그 피처들은 언제든지 임의로 제거될 수 있기 때문이다.
따라서 모든 피처들이 균일한 비중을 갖게 spread out해야 할 것이다.
이것은 w의 제곱 노름을 축소시키는 효과가 있다.
그러므로 dropout은 l2 일반화와 비슷하게 과적합을 막아준다.
dropout은 l2 일반화의 변형된 버전이라 할 수 있다.




8. Other regularization methods
Data augmentation
 사진일 경우, 가로로 뒤집거나 사진을 줌인하는 등
 가짜로 추가적인 샘플을 만들 수 있다. 
 그러나 성능은 새로운 고양이 사진보다 낮을 것이다.

Early stopping
 기울기 강하를 실행하면서 트레이닝 오류나 최적화 J 함수를 그린 뒤,
 dev set error의 그래프를 그려보자.
 dev error는 계속 감소하다가 어느 지점 이후로 다시 증가하는 것을 볼 수 있다.
 그 부분을 기점으로 신경망의 성능이 떨어졌기 때문에 
 early stopping은 거기까지만 진행될 수 있도록 멈춰준다.
 결과적으로 신경망이 과적합되는 것을 막아준다.
 
 단점으로는 머신러닝의 주요한 2단계, 
 즉 J를 최적화 하는것과 과적합되지 않게 만드는 것 두 별개의 문제를
 단독으로 풀 수 없게 만든다는 것이다.
 기울기 강화를 일찍 정지시키기 때문에 비용함수 J를 최적화시키는 도중에 멈추게 만들고,
 과적합되지 않도록 하기 때문에 두 가지 문제가 약간씩 섞인다.
 
 early stoppin 대신에 l2 일반화를 사용하는 방법도 있는데,
 이 경우 신경망을 최대한 길게 훈련시키면 되지만
 여러가지 일반화 파라미터 lambda를 시도해봐야하는 단점이 있다.





9. Normalizing inputs
신경망을 훈련시킬 때, 트레이닝의 속도를 높일수 있는 방법은 표준화이다.
입력 피처 = x1, x2

입력값을 표준화 시키는 절차.
 1) 평균값을 빼거나 0으로 만든다. mu = (1 / m) * sum(x)
 2) 변동 편차를 표준화 한다. sigma = (1 / m) * sum(x ** 2)
 표준화 할 때, 훈련세트와 테스트세트를 동일하게 표준화하는 것이 좋다.

또, 입력피처의 범위가 크게 다를경우
ex) x1 = 1 ~ 100000,  x2 = 0 ~ 1
authorization 알고리즘에 문제가 생긴다.
따라서 모든것을 평균값 0으로 지정하고 편차가 1이 되도록
표준화해주면 잘 작동하게 된며 학습 속도도 빨라진다.




10. Vanishing / Exploding gradients
신경망을 교육할 때 생기는 가장 큰 문제점은
 1. 데이터가 사라지는 경우 
 2. 기울기가 폭발적인 증가를 하는 경우 이다.
심층신경망을 학습할 때 derivatives나 기울기가 굉장히 크거나 작은 값을 가질 때,
트레이닝이 매우 까다로울 수 있다.

심층망이 깊어질수록
다른 모든 것들이 1보다 조금 큰 값이라고 할 때,
W > I 라면 기울기가 기하급수적으로 커지고
W < I 라면 기울기가 기하급수적으로 작아진다.




11. Numerical approximation of gradients
gradient checking할 때는 one-sided difference보다
two-sided difference공식이 더 정확하다.




12. Gradient Checking Implementation Notes
 1. 트레이닝할 때는 grad check을 사용하면 안된다.
    오로지 debug할 때만 사용하자.
 2. 만약 알고리즘이 grad check에 실패할 경우, 
    요소를 살펴보고 버그를 찾아야한다.
 3. 일반화 할 때는 일반화항을 기억해두어야 한다.
 4. grad check는 dropout에서는 실행되지 않는다.

    



"""