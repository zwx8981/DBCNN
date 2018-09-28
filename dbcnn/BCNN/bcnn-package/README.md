# Bilinear Pooling: Package extension of MatConvNet

Created by Tsung-Yu Lin, Aruni RoyChowdhury and Subhransu Maji at UMass Amherst

### Introduction

This repository contains the code implemting bilinear pooling proposed in our ICCV 2015 paper:

	@inproceedings{lin2015bilinear,
        Author = {Tsung-Yu Lin, Aruni RoyChowdhury, and Subhransu Maji},
        Title = {Bilinear CNNs for Fine-grained Visual Recognition},
        Booktitle = {International Conference on Computer Vision (ICCV)},
        Year = {2015}
    }
	
The code is tested on Ubuntu 14.04 using NVIDIA K40 GPU and MATLAB R2014b.

Link to the [project page](http://vis-www.cs.umass.edu/bcnn).

### Installation
This code is the extension to support bilinear pooling on [MatConvNet](http://www.vlfeat.org/matconvnet).Follow instructions on MatConvNet pages to install it first. Our code is built on MatConvNet version `1.0-beta18`. To retrieve a particular version of MatConvNet using git type:

	>> git fetch --tags
	>> git checkout tags/v1.0-beta18
      
The code implements the bilinear combination layer in symmetic and assymetic CNNs and contains scripts to fine-tune models and run experiments on several fine-grained recognition datasets. We also provide pre-trained models.

### Usage
The package provides following functions:

For simplenn:
   
1. **vl_nnbilinearpool**

   The implementation of pooling self-outer product across locations to get an image descriptor. The function is called in *vl_bilinearnn* when the layer has field `type='bilinearpool'`.
   
2. **vl_nnbilinearclpool**

   The function takes two features maps x1 and x2 as input and computer the outer product between features which are pooled across locations to form an image descriptor. The function is called in *vl_bilinearnn* when the layer has field `type='bilinearclpool'`.
   
3. **vl_nnsqrt**

   The function normalize features using square root normalization for every location. The function is called in *vl_bilinearnn* when the layer has field `type='sqrt'`.

4. **vl_nnl2norm**: 

   The function normalize features using l2 normalization for every location. The function is called in *vl_bilinearnn* when the layer has field `type='l2norm'`.

5. **vl_bilinearnn**: Similar to vl_simplenn but with support of above layers. The function is used to compute forward and backward passes through a netwrok.

For DagNN:

1. **BilinearClPooling**

   Dagnn bilinear pooling layer taking two set of features as input and computing the outer product followed by sum pooling.
    
2. **BilinearPooling**

   Dagnn bilinear pooling layer taking features as input and computing the self outer product followed by sum pooling.
   
3. SquareRoot

   Dagnn square root normalization layer

4. L2Norm

   Dagnn L2 normalization layer




### Acknowldgements

We thank MatConvNet and VLFEAT teams for creating and maintaining these excellent packages.