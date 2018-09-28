% setup vlfeat
run ./vlfeat/toolbox/vl_setup

% setup matconvnet
run ./matconvnet/matlab/vl_setupnn
addpath ./matconvnet/examples/

% add bcnn package
addpath ./bcnn-package/

clear mex ;
