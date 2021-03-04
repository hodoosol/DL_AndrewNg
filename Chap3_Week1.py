"""
2021.02.22
Andrew Ng - Deep Learning
Chater 2. Structuring Machine Learning Projects
Week 1. ML Strategy(1)

"""



"""
1. Why ML Strategy
In Cat Classifier, 정확도가 90%라면
  1) 더 많은 데이터 수집
  2) 다양한 트레이닝 세트 수집
  3) 경사하강법으로 더 길게 알고리즘 훈련
  4) 경사하강법 대신 아담으로 훈련
  5) 더 큰 네트워크, 더 작은 네트워크 시도
  6) dropout 시도
  7) L2 표준화 시도
  8) 활성화 함수를 바꾸거나 숨은 유닛의 개수를 바꾸는 등
여러가지 방법으로 정확도를 올릴 수 있다.
그러나 이 모든 방법을 하나하나 시도하다간 시간이 훌쩍 지나버릴 것이다.
따라서 ML 전략을 사용하여 어떤 방법을 선택하고 버릴지 알아보자.





2. Orthogonalization
머신러닝에서는, 어떤 파라미터를 튜닝할 건지 골라내는 것이 중요하다.

지도학습이 잘 진행되기 위해서는
  1) 훈련세트에서의 성능이 어느정도 좋아야 한다.
     -> 더 큰 신경망, 다른 최적화 알고리즘 시도
  2) dev 세트에서도 good
     -> 정규화, 더 큰 트레이닝 세트 적용
  3) 테스트세트에서도 good
     -> 더 큰 dev 세트 적용
  4) 비용함수가 적용된 실제 데이터에서도 잘 작동해야한다.
     -> dev 세트나 비용함수 변경





3. Single number evaluation metric
precision(정밀도) - 샘플의 몇 퍼센트를 고양이로 인식했는가 ?
recall(재현율) - 실제 몇 퍼센트를 고양이로 인식했는가 ?
각각의 정밀도와 재현율이
A - 95%, 90%
B - 98%, 85% 라고 할 때 어떤 모델이 더 좋은지 고르긴 어렵다.
따라서 정밀도와 재현율을 결합시킨 새로운 평가 지표를 찾아야 한다.
F1 score = 정밀도와 재현율 값의 평균 수치 (Harmonic mean)





4. Satisficing and Optimizing metric
In Cat classification example, 
정확도와 running time 두 개의 지표로 성능을 따져보자.

a - 90%, 80ms
b - 92%, 95ms
c - 95%, 1500ms

cost = accuracy - 0.5 * running time 으로 따져볼 수도 있지만

정확도는 maximize하고, 러닝타임은 100ms 이하라면 통과하도록 해본다면
정확도는 optimizing, 러닝타임은 satisficing metric이 된다.
그렇다면 가장 성능이 좋은 모델은 b모델이 될 것이다.





5. Train/dev/test distributions
dev와 test 세트를 설정하는 방법

dev와 test 세트가 모두 다른 분포도를 갖으면 x
무작위로 섞은 데이터를 dev와 테스트세트에 반영하여 동일한 분포도를 가지게끔 설정해야 한다.

따라서 처음 dev와 테스트 세트를 설정할 때, 
미래에 어떤 데이터를 반영할지와 어떤 것을 중요시할지를 고려해야 한다.





6. Size of the dev and test sets
최근, 데이터세트의 크기가 커지면서
기존의 train, dev, test 세트를 나누던 비율은 비효율적이게 되었다.
따라서 만약 100만개의 데이터세트가 있다면
98%를 트레이닝세트로, 각각 1%를 dev와 테스트세트에 할당하는 것이 합리적이다.





7. When to change dev/test sets and metrics
만약 평가 메트릭이 원하는/적절한 순서로 선호도를 알려주지 않는다면
ex) a - 3% 에러, but porno 포함
    b - 5% 에러, porno 포함 X
유저입장에서는 b 모델이 더 좋을 것이나, 성능이 더 높은 a를 고른다면
새로운 평가 메트릭을 도입해야할 것이다.


ex) a - 3% 에러
    b - 5% 에러
먄약 dev 세트에서 a가 b보다 더 성능이 좋지만
실제 사용자들이 올리는 고양이의 사진은 흐릿하고, 여러 변수가 있기 때문에
결국 b 알고리즘이 더 잘 작동한다면 ?
역시 메트릭과 dev/test 세트를 변경해야 할 것이다.





8. Why human-level performance? 
인간 레벨의 성능을 고려하는 이유는 ?
  1) 딥러닝의 발전으로 머신러닝이 갑자기 더 잘 구동되어서
  2) 머신러닝을 디자인하고 수행 절차를 수립하는 과정이 효율적으로 발전해서

알고리즘을 학습시키면 학습시킬수록 인간레벨의 성능을 뛰어 넘으면서
진행속도나 정확도가 상승하는 정도는 보통 더뎌지게 된다.
그러나 아무리 모델을 확장시키거나 더 많은 데이터를 수집한다해도
이론적으로 최적화된 성능레벨에 도달하진 못한다.
이것을 Bayes optimal error라고 한다.

그렇다면 왜 인간레벨 성능을 뛰어넘은 뒤 진행속도가 더뎌질까 ?
  1) 많은 수행업무에 있어서 인간레벨 성능은 bayes error와 크게 떨어져있지 않아서
     그 이상 발전할 수 있는 부분이 제한적이기 때문이다.
  2) 인간레벨 성능에 미치지 못할 때, 사용할 수 있는 툴은 아주 많지만
     초과하는 시점에선 사용할 수 없게 된다.






9. Avoidable bias
인간 error - 1%
training error - 8%
dev error - 10% 라면
해당 알고리즘은 트레이닝 세트에서 잘 피팅되지 않는 것이다.
그러므로, 바이어스를 줄여야할 것이다.

인간 error - 7.5%
training error - 8%
dev error - 10% 라면
해당 알고리즘은 트레이닝 세트에서는 잘 하고 있다.
따라서 dev의 오류를 좀 더 줄이기 위해 
일반화시키거나 트레이닝 데이터를 더 수집하여
러닝알고리즘의 편차를 줄여야 한다.

avoidable bias라는 용어는 
최소의 오류값이나 특정 오류가 있다는 것을 인정하는 것이고
bayes  error가 7.5%일 때, 이 이하로 내려갈 수 없다는 뜻을 내포한다.
따라서 1번째 예시의 avoidable bias는 7% 이고 편차는 2%,
2번째 예시의 avoidable bias는 0.5% 이고 편차는 2%이다.





7. Understanding human-level performance
human-level performance란 ?
의학 사진을 분류할 때,
 a) 특정 인간 - 3% 오류
 b) 특정 의사 - 1% 오류
 c) 숙련된 의사 - 0.7% 오류
 d) 숙련된 의사들의 팀 - 0.5% 오류 라고 해보자.
이 중 어떤 것이 human-level performance일까 ?
이 경우에서는 d의 0.5%를 bayes error의 추정치로 설정한다.

따라서 human-level performance는 0.5% 이고
training error - 5%
dev error - 6% 일 때,
avoidable bias는 4퍼센트 정도이고 편차는 1퍼센트일 것이다.

그러나 학습을 계속하여 0.5% / 0.7% / 0.8% 와 같이 인간 성능에 가까워질 때,
바이어스와 편차 효과를 제거하기가 점점 더 어려워진다.





8. Surpassing human-level performance
팀 - 0.5% error
인간 1명 - 1% error
training - 0.6% error
dev - 0.8% error    일 때,

avoidable bias는 ?
팀에서 트레이닝 세트의 에러를 뺀 0.1%
편차는 ? 0.2% 


팀 - 0.5% error
인간 1명 - 1% error
training - 0.3% error
dev - 0.4% error    일 때,

avoidable bias는 ?
정보가 충분하지 않기 때문에 알고리즘에서 bias를 줄이는데 중점을 두어야 할지,
편차를 줄이는데 중점을 두어야할지 애매하여 대답하기 힘들다.
또, 트레이닝 세트의 에러가 이미 인간의 팀보다 좋다고 하면
인간의 직관적인 부분에 의존하기 어려워진다.
해당 0.5%의 한계치(인간팀 에러)를 능가했을 때, 
머신러닝 알고리즘의 진행 및 발전 방법이 명백하지 않아진다.

요즘의 ML 알고리즘은 인간성능 레벨을 뛰어넘은 것이 많아졌다.
ex) 온라인 광고, 제품 추천, 운송 요금 예측, 대출 승인 등
이 네가지의 예들은 
이용자들이 클릭했던 내역, 그동안 이루어진 운송 요금 데이터, 대출 신청내역과 결과 등의
구조화된 데이터로 학습 가능하다는 공통점이 있다.





9. Improving your model performance
효과적인 학습 알고리즘을 위해서는 두가지 전제조건이 있다.
  1) 트레이닝 세트를 잘 대입할 수 있다.
      ->  낮은 avoidable bias를 획득할 수 있다.
  2) dev/test 세트에서도 잘 대입할 수 있다.
      -> 낮은 편차를 획득할 수 있다.

낮은 avoidable bias를 획득하기 위해서는 
더 큰 모델을 트레이닝 시키거나
트레이닝 세트에서 더 잘, 길게 학습시켜야 한다.(더욱 향상된 알고리즘 사용)

낮은 편차를 획득하기 위해서는
더 많은 데이터를 수집하거나
일반화, dropout, data augmentation 등을 시도해봐야 한다.





"""