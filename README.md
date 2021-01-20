# ST-AMCNN:	Pose-Guided Matching for Assessing Quality of Action on Rehabilitation Training

By Yuhang Qiu, Jiping Wang, Zhe Jin, Mingliang Zhang and Liquan Guo at Chinese Academy of Sciences and Monash university.

### Introduction

**ST-AMCNN** is a patch-based Siamese Convolutional Neural Network employed for pose-guided matching on the eight-section brocade dataset which is one of the most representative traditional rehabiblitation exercises in China.  

Here is a the test model for demonstration. This code is implemented by Python. When the paper is received, the complete code will be made public

### Requirements: software

0. `Python 3.7`: cv2, numpy, scipy, matplotlib, PIL, sklearn
0. `Pytorch 1.5.1`

### Predicting Demo
0.  For similar pose images matching : Run `python ST-AMCNN.py 1 2` or `python ST-AMCNN.py 3 4` to test demo images provided in `images/`.
    - You will see the visualized comparsion results, The redder areas indicated this part of poses is highly matched and obtains high score between the learner’s pose and         standard pose.
0.  For different pose images matching : Run `python ST-AMCNN.py 1 3` or `python ST-AMCNN.py 2 4` to test demo images provided in `images/`.
    - You will see there are few comparative results because the input poses are totally different. The model cannot give the suggestion for rectification.


