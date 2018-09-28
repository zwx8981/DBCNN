function y = vl_nnsqrt(x, param, varargin)
% VL_NNSQRT perform square root normalization for the input features
% at each location
%
% Author: Subhransu Maji, Aruni RoyChowdhury, Tsung-Yu Lin

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% input:
% forward pass:
% x: the features of size [hight, width, channels, batches]
% param: the threshold to prevent large value when close to 0
% backward pass:
% x: the features of size [hight, width, channels, batches]
% param: the threshold to prevent large value when close to 0
% dzdy: the gradient with respect to output y

% output:
% forward pass:
% y: taking the element-wise singed square root of x. Output size is the
%    same as input
% backward pass:
% y: the gradient with respect to x

thresh = param(1);

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

if backMode
    y = 0.5./sqrt(abs(x)+thresh);
    y = y.*dzdy;
else
    y = sign(x).*sqrt(abs(x));
end
