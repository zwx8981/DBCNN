
Deep Bilinear Pooling for Blind Image Quality Assessment 
=
Weixia Zhang, Kede Ma, Jia Yan, Dexiang Deng, and Zhou Wang
-
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), to appear, 2019.
-


Files under distorion_generator are used for synthetizing distorted images.
-

Usuage:

[ distorted_img, map ] = distortion_generator( img, dist_type, level, seed )

Where img is the original pristine image, dist_type refers to a specified distortion type ranging in 1~9.

1, Gaussian Blur \
2, White Noise  \
3, JPEG Compression \
4, JPEG2000 Compression \
5, Contrast Change \
6, Pink Noise \
7, Image Color Quantization with Dither \
8, Overexposure \
9, Underexposure 

level is a specified degradation level range in 1~5.

seed should be fixed to be 1.

Distorted_img is the output distored image, map is only used for saving distorted images of the type Image Color Quantization with Dither. 


Training codes live in dbcnn folder.
-

Running the run_exp.m script to train and test on a specifid dataset across 10 random splits.

Prerequisite: Matlab(We use 2017a), MatConvNet (We use 1.0-beta25)， vlfeat(We use 0.9.2)

Pretrained s-cnn model is included in dbcnn\data\models, you should download vgg-16 model from http://www.vlfeat.org/matconvnet/pretrained/ and put it in dbcnn\data\models.


Relevant links: \
Waterloo Exploration Database: https://ece.uwaterloo.ca/~k29ma/exploration/ \
PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
