# BIQA_project

Files under distorion_generator are used for synthetizing distorted images.

Usuage:

[ distorted_img, map ] = distortion_generator( img, dist_type, level, seed )

Where img is the original pristine image, dist_type refer to a specified distortion type ranging in 1~9.

1, Gaussian Blur
2, White Noise
3, JPEG Compression
4, JPEG2000 Compression
5, Contrast Change
6, Pink Noise
7, Image Color Quantization with Dither
8, Overexposure
9, Underexposure

level is a specified degradation level range in 1~5.

seed is fixed to be 1.

Distorted_img is the output distored image, map is only used for saving distorted images of the type Image Color Quantization with Dither.
