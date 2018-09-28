function y = vl_nnbilinearpool(x, varargin)
% VL_NNBILINEARPOOL computes self outer product of a feature x and pool the features across 
% all locations
%
% Author: Subhransu Maji, Aruni RoyChowdhury, Tsung-Yu Lin

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% input:
% forward pass:
% x: input featre of size [hight, width, channels, batches]
% backward pass:
% x: input featre of size [hight, width, channels, batches]
% dzdy: the gradient with respect to output y.


% output:
% forward pass:
% y: self outer product of x.Features are pooled across locations
%    The output size if [1, 1, channels*channels, batches]
% backward pass:
% y: graident with respect to x. y will have the same size as input x.

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
end

gpuMode = isa(x, 'gpuArray');
[h, w, ch, bs] = size(x);

if backMode
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch, ch]);
        a = reshape(x(:,:,:,b), [h*w, ch]);
        y(:, :, :, b) = reshape(a*dzdy_b, [h, w, ch])/(h*w);
    end
else
    if gpuMode
        y = gpuArray(zeros([1, 1, ch*ch, bs], 'single'));
    else
        y = zeros([1, 1, ch*ch, bs], 'single');
    end
    for b = 1:bs,
        a = reshape(x(:,:,:,b), [h*w, ch]);
        y(1,1,:, b) = reshape(a'*a, [1 ch*ch])/(h*w);
    end
end

