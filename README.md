# facerecog

  Google Cloud platform Vision API를 통한 face detection과 자체 실험을 통해 제작한 구조를 통하여 classification을 실행한 얼굴 인식 시스템 제작
  (http://bcho.tistory.com/1178?category=555440 참고함)
  
# 특징
  1. 96x96x3의 이미지를 입력으로 사용한다.
  <p align="center"><img src="https://user-images.githubusercontent.com/33644885/103069846-998e0600-4603-11eb-9448-8cf0851129ed.png"></img></p>
  2. 위와 같은 자체 개발 구조를 사용한다.<p></p>
  3. AlexNet에서 사용한 overlapped pooling, dropout을 사용하여 overfitting을 감소시켰다.
  4. Xavier initializer 를 사용하였다.
  
  
# 장점
  1. 이것저것 많이 사용하여 학습의 효율이 높다. 
  2. 높은 성능을 보인다

#단점
  1. 이것저것 많이 사용하여 파라미터가 많다. 즉 학습이 느리다.
  
  
#마치며
  인공지능 처음 공부해봤는데 재미있었다. 허허껄껄
