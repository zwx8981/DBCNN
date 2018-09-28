function [y, varargout] = vl_nnbilinearclpool(x1, x2, varargin)
% VL_NNBILINEARPOOL computes outer product of x1 and x2, and pool the features 
% across all locations

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


% input:
% forward pass:
% x1: first input featre of size [h1, w1, ch1, batches]
% x2: second input featre of size [h2, w2, ch2, batches]
% backward pass:
% x1: first input featre of size [h1, w1, ch1, batches]
% x2: second input featre of size [h2, w2, ch2, batches]
% dzdy: the gradient with respect to output y.

% output:
% forward pass:
% y: outer product between x1 and x2. The features are pooled across locations. 
%    The output size if [1, 1, ch1*ch2, batches]
% backward pass:
% y: graident with respect to x1. y1 will have the same size as input x1.
% y2: varargout{1} graident with respect to x2. y2 will have the same size
%     as input x2


% flag for doing backward pass
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end

% if GPU is used
gpuMode = isa(x1, 'gpuArray');

% [height width channels batchsize]
[h1, w1, ch1, bs] = size(x1);
[h2, w2, ch2, ~] = size(x2);



% resize the convolutional output to the same resolution
if w1*h1 <= w2*h2,
    %downsample feature 2
    x2 = array_resize(x2, w1, h1);
else
    %downsample feature 1
    x1 = array_resize(x1, w2, h2);
end
h = size(x1, 1); w = size(x1, 2);




if backMode
    % do backward pass
    if gpuMode
        y1 = gpuArray(zeros(h1, w1, ch1, bs, 'single'));
        y2 = gpuArray(zeros(h2, w2, ch2, bs, 'single'));
    else
        y1 = zeros(h1, w1, ch1, bs, 'single');
        y2 = zeros(h2, w2, ch2, bs, 'single');
    end
    
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch1, ch2]);
        A = reshape(x1(:,:,:,b), [h*w, ch1]);
        B = reshape(x2(:,:,:,b), [h*w, ch2]);
        dB = reshape(A*dzdy_b, [h, w, ch2]);
        dA = reshape(B*dzdy_b', [h, w, ch1]);
        
        if w1*h1 <= w2*h2
            %B is downsampled
            indw = round(linspace(1,w2,w1));
            indh = round(linspace(1,h2,h1));
            y2(indh,indw,:,b) = dB;
            y1(:,:,:,b) = dA;
        else
            %A is downsampled
            indw = round(linspace(1,w1,w2));
            indh = round(linspace(1,h1,h2));
            y2(:,:,:,b) = dB;
            y1(indh,indw,:,b) = dA;
        end
    end
    y = y1;
    varargout{1} = y2;
else
    % do forward pass
    if gpuMode
        y = gpuArray(zeros([1, 1, ch1*ch2, bs], 'single'));
    else
        y = zeros([1, 1, ch1*ch2, bs], 'single');
    end
    
    for b = 1:bs,
                
        xa = reshape(x1(:,:,:,b), [h*w, ch1]);
        xb = reshape(x2(:,:,:,b), [h*w, ch2]);
        
        y(1,1,:, b) = reshape(xa'*xb, [1 ch1*ch2]);
    end
end





function Ar = array_resize(A, w, h)
    indw = round(linspace(1,size(A,2),w));
    indh = round(linspace(1,size(A,1),h));
    Ar = A(indh, indw, :, :);
