An experimental PyTorch implementation of DB-CNN is released at https://github.com/zwx8981/DBCNN-PyTorch! 


Blind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network (Official IEEE version)
=
Weixia Zhang, Kede Ma, Jia Yan, Dexiang Deng, and Zhou Wang
https://ieeexplore.ieee.org/document/8576582

IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Volume: 30 , Issue: 1 , Jan. 2020.


Files under distorion_generator are used for synthesizing distorted images.
-

Usuage:

distorted_img = distortion_generator( img, dist_type, level, seed )

Where img is the original pristine image, dist_type refers to a specified distortion type ranging in 1~9.

1, Gaussian Blur \
2, White Noise  \
3, JPEG Compression \
4, JPEG2000 Compression \
5, Contrast Change \
6, Pink Noise \
7, Image Color Quantization with Dither \
8, Over-Exposure \
9, Under-Exposure 

level is a specified degradation level range in 1~5.

seed should be fixed to be 1.


Training codes live in dbcnn folder.
-

Running the run_exp.m script to train and test on a specifid dataset across 10 random splits.

Prerequisite: Matlab(We use 2017a), MatConvNet (We use 1.0-beta25)， vlfeat(We use 0.9.2)

Pretrained s-cnn model is included in dbcnn\data\models, you should download vgg-16 model from http://www.vlfeat.org/matconvnet/pretrained/ and put it in dbcnn\data\models.

You need to copy the matconvet/matlab folder to that of your matconvnet to modify the vl_simplenn.m and PDist.m files. 
-

Relevant links: \
Waterloo Exploration Database: https://ece.uwaterloo.ca/~k29ma/exploration/ \
PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

Citation
-
@article{zhang2020blind,
  title={Blind Image Quality Assessment Using A Deep Bilinear Convolutional Neural Network},
  author={Zhang, Weixia and Ma, Kede and Yan, Jia and Deng, Dexiang and Wang, Zhou},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={30},
  number={1},
  pages={36--47},
  year={2020}
}
