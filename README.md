# BIQA_project

1, Files under distorion_generator are used for synthetizing distorted images.\

Usuage:\

[ distorted_img, map ] = distortion_generator( img, dist_type, level, seed )\<br>  

Where img is the original pristine image, dist_type refer to a specified distortion type ranging in 1~9.\<br>  

1, Gaussian Blur\
2, White Noise\<br>  
3, JPEG Compression\<br>  
4, JPEG2000 Compression\<br>  
5, Contrast Change\<br>  
6, Pink Noise\<br>  
7, Image Color Quantization with Dither\<br>  
8, Overexposure\<br>  
9, Underexposure\<br>  

level is a specified degradation level range in 1~5.\<br>  

seed is fixed to be 1.\<br>  

Distorted_img is the output distored image, map is only used for saving distorted images of the type Image Color Quantization with Dither.\<br>  

2, Files under train_test_split are imdb files that record the specified training and testing splits used in the paper on four IQA benchmark databases, LIVE IQA, CSIQ, TID2013 and LIVE Challenge\<br>  
