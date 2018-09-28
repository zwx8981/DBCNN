function y = vl_nnbilinear(x, varargin)
% VL_NNBILINEAR  extracts bilinear features at each location
% This function differs from vl_nnbilinerpool whichs pools features across
% locations

% input:
% forward pass:
% x: input featre of size [hight, width, channels, batches]
% backward pass:
% x: input featre of size [hight, width, channels, batches]
% dzdy: the gradient with respect to output y.

% output:
% forward pass:
% y: self outer product of x at each pixel location. The output size if [hight, width, channels*channels, batches]
% backward pass:
% y: graident with respect to x. y will have the same size as input x.

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

[h, w, ch, bs] = size(x);
gpuMode = isa(x, 'gpuArray');

if backMode
    
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    for b=1:bs
        for yy=1:h
            for xx=1:w
                dzdy_b = reshape(dzdy(yy,xx,:,b), [ch, ch]);
                a = squeeze(x(yy,xx,:,b));
                y(yy, xx, :, b) = reshape(a'*dzdy_b, [h, w, ch]);
            end
        end
    end
else
    
    if gpuMode
        y = gpuArray(zeros([h, w, ch*ch, bs], 'single'));
    else
        y = zeros([h, w, ch*ch, bs], 'single');
    end
    for b = 1:bs,
        for yy = 1:h,
            for xx = 1:w,
                a = squeeze(x(yy,xx,:,b));
                y(yy,xx,:, b) = reshape(a*a', [1 ch*ch]);
            end
        end
    end
end
