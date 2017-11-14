# BIQA_project


1, Files under distorion_generator are used for synthetizing distorted images.

Usuage:

[ distorted_img, map ] = distortion_generator( img, dist_type, level, seed )

Where img is the original pristine image, dist_type refer to a specified distortion type ranging in 1~9.

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

seed is fixed to be 1.

Distorted_img is the output distored image, map is only used for saving distorted images of the type Image Color Quantization with Dither. 

2, Files under train_test_split are imdb files that record the specified training and testing splits used in the paper on four IQA benchmark databases, LIVE IQA, CSIQ, TID2013 and LIVE Challenge. Image names, labels(subjective quality scores) and set (1 indicates training and 2 indicated testing) are stored in imdb.images.name, imdb.images.label, imdb.images.set respectively. \

3. Files under pre-trained mdoel is the shallow CNN pre-trained on the sythetically distorted images set. It has already been tailored as described in the paper. 

The training code will be released in the future, however, you can also do it yourself following this instruction:

1, Prerequisite: Matlab(We use 2017a), MatConvNet (We use 1.0-beta25).\
2, Get BCNN-package and example training code from https://bitbucket.org/tsungyu/bcnn-package and https://bitbucket.org/tsungyu/bcnn.git respectively. \
3, Example B-CNN codes are used for training models for image classification, so you need to modify the file initializeNetworkTwoStreams.m by replace the softmax layer by L2-Loss layer, which is used for regression task. You also need to modify imdb_get_batch_bcnn.m to make sure images are fed into the networks during training in appropriate ways. (randomly crop 432X432 patches from original images on LIVE IQA and fed with original size on other three databases, as described in the paper.)Actually, you can direclty implement L2-LOSS using vl_nnpdist function which is included in MatConvNet by setting 'aggregate' option to True.\
4, Modify model_setup.m and run_experiments_bcnn_train.m such that they meet your requirements.

Relevant links: \
Waterloo Exploration Database: https://ece.uwaterloo.ca/~k29ma/exploration/ \
PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
