function y = vl_nnbilinearcl(x1, x2,  varargin)
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


gpuMode = isa(x1, 'gpuArray');

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
    
    if gpuMode
        y1 = gpuArray(zeros(h1, w1, ch1, bs, 'single'));
        y2 = gpuArray(zeros(h2, w2, ch2, bs, 'single'));
    else
        y1 = zeros(size(h1, w1, ch1, bs, 'single'));
        y2 = zeros(size(h2, w2, ch2, bs, 'single'));
    end
    for b=1:bs
        dzdy_b = reshape(dzdy(1,1,:,b), [ch1, ch2]);
        dA = zeros(h, w, ch1, 'single');
        dB = zeros(h, w, ch2, 'single');

        %% TO DO
        for yy=1:h
            for xx=1:w
                dzdy_b = reshape(dzdy(yy,xx,:,b), [ch1, ch2]);
                a = squeeze(x1(yy,xx,:,b));
                b = squeeze(x2(yy,xx,:,b));
                dB(yy, xx, b) = a'*dzdy_b;
                dA(yy, xx, b) = b'*dzdy_b';

            end
        end
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
    
    if gpuMode
        y = gpuArray(zeros([h, w, ch1*ch2, bs], 'single'));
    else
        y = zeros([h, w, ch1*ch2, bs], 'single');
    end
    for b = 1:bs,
        for yy = 1:h,
            for xx = 1:w,
                xa = squeeze(x1(yy,xx,:,b));
                xb = squeeze(x2(yy,xx,:,b));
                y(yy,xx,:, b) = reshape(xa*xb', [1 ch1*ch2]);
            end
        end
    end
end


function Ar = array_resize(A, w, h)
    indw = round(linspace(1,size(A,2),w));
    indh = round(linspace(1,size(A,1),h));
    Ar = A(indw, indh, :, :);
