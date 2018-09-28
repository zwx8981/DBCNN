function res = vl_bilinearnn(net, x, dzdy, res, varargin)
% VL_BILINEARNN is the extension of VL_SIMPLENN to suppport 
%        1.vl_nnbilinearpool()
%        2.vl_nnbilinearclpool()
%        3.vl_nnsqrt()
%        4.vl_nnl2norm()
%   RES = VL_BILINEARENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_BILINEARNN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%
%   The network has a simple (linear) topology, i.e. the computational
%   blocks are arranged in a sequence of layers. Please note that
%   there is no need to use this wrapper, which is provided for
%   convenience. Instead, the individual CNN computational blocks can
%   be evaluated directly, making it possible to create significantly
%   more complex topologies, and in general allowing greater
%   flexibility.
%   
%   The NET structure contains two fields:
%
%   - net.layers: the CNN layers.
%   - net.normalization: information on how to normalize input data.
%
%   The network expects the data X to be already normalized. This
%   usually involves rescaling the input image(s) and subtracting a
%   mean.
%
%   RES is a structure array with one element per network layer plus
%   one representing the input. So RES(1) refers to the zeroth-layer
%   (input), RES(2) refers to the first layer, etc. Each entry has
%   fields:
%
%   - res(i+1).x: the output of layer i. Hence res(1).x is the network
%     input.
%
%   - res(i+1).aux: auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - res(i+1).dzdx: the derivative of the network output relative to
%     variable res(i+1).x, i.e. the output of layer i. In particular
%     res(1).dzdx is the derivative of the network output with respect
%     to the network input.
%
%   - res(i+1).dzdw: the derivative of the network output relative to
%     the parameters of layer i. It can be a cell array for multiple
%     parameters.
%
%   net.layers is a cell array of network layers. The following
%   layers, encapsulating corresponding functions in the toolbox, are
%   supported:
%
%   Convolutional layer::
%     The convolutional layer wraps VL_NNCONV(). It has fields:
%
%     - layer.type = 'conv'
%     - layer.filters: the filters.
%     - layer.biases: the biases.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - layer.type = 'pool'
%     - layer.method: pooling method ('max' or 'avg').
%     - layer.pool: the pooling size.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.padding: the padding (usually 0).
%
%   Normalization layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
%   ReLU layer::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%     - layer.class: the ground-truth class.
%
%   Softmax-log-loss layer::
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%     - layer.class: the ground-truth class.
%
%   Bilinear-pool layer::
%       The bilinear-pool layer wraps VL_NNBILINEARPOOL(). It has fields:
%
%       - layer.type = 'bilinearpool'
% 
%   Bilinear-cross-layer-pool layer::
%       The bilinear-cross-layer-pool layer wraps VL_NNBILINEARCLPOOL(). It has fields:
%
%       - layer.type = 'bilinearclpool'
%       - layer.layer1: one input from the output of layer1
%       - layer.layer2: one input from the output of layer2
% 
%   Square-root layer::
%       The square-root layer wraps VL_NNSQRT(). It has fields:
%
%       - layer.type = 'sqrt'
%
%   L2 normalization layer::
%       The l2 normalization layer wraps VL_NNL2NORM(). It has fields:
%
%       - layer.type = 'l2norm'
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - layer.type = 'custom'
%     - layer.forward: a function handle computing the block.
%     - layer.backward: a function handle computing the block derivative.
%
%     The first function is called as res(i+1) = forward(layer, res(i), res(i+1))
%     where res() is the struct array specified before. The second function is
%     called as res(i) = backward(layer, res(i), res(i+1)). Note that the
%     `layer` structure can contain additional fields if needed.
%
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN. It is modified from VL_SIMPLENN.m of MATCONVNET package
% and is made available under the terms of the BSD license (see the COPYING file).

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.mode = 'normal' ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;
% opts.doforward = true;
opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;


if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end


if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end


switch lower(opts.mode)
  case 'normal'
    testMode = false ;
  case 'test'
    testMode = true ;
  otherwise
    error('Unknown mode ''%s''.', opts. mode) ;
end

docrosslayer = false;
for i=1:n
  l = net.layers{i} ;
  if(strcmp(l.type, 'bilinearclpool'))
      docrosslayer = true;
      crlayer1 = l.layer1;
      crlayer2 = l.layer2;
  end
end


gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
res(1).x = x ;





for i=1:n
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      res(i+1).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
        'pad', l.pad, ...
        'stride', l.stride, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'convt'
      res(i+1).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
        'crop', l.crop, ...
        'upsample', l.upsample, ...
        'numGroups', l.numGroups, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case 'pool'
      res(i+1).x = vl_nnpool(res(i).x, l.pool, ...
        'pad', l.pad, 'stride', l.stride, ...
        'method', l.method, ...
        l.opts{:}, ...
        cudnn{:}) ;
    case {'normalize', 'lrn'}
      res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i+1).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i+1).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
      res(i+1).x = vl_nnrelu(res(i).x,[],leak{:}) ;
    case 'sigmoid'
      res(i+1).x = vl_nnsigmoid(res(i).x) ;
    case 'noffset'
      res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'spnorm'
      res(i+1).x = vl_nnspnorm(res(i).x, l.param) ;
    case 'dropout'
      if testMode
        res(i+1).x = res(i).x ;
      else
        [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
      end
    case 'bnorm'
      if testMode
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, 'moments', l.weights{3}) ;
      else
        res(i+1).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
      end
    case 'pdist'
      res(i+1) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
    case 'bilinearpool'
      res(i+1).x = vl_nnbilinearpool(res(i).x);
    case 'bilinearclpool'
      x1 = res(l.layer1+1).x;
      x2 = res(l.layer2+1).x;
      res(i+1).x = vl_nnbilinearclpool(x1, x2);
    case 'sqrt'
      res(i+1).x = vl_nnsqrt(res(i).x, 1e-8);
    case 'l2norm'
      res(i+1).x = vl_nnl2norm(res(i).x, 1e-10);
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  forget = forget & (~docrosslayer || (i~=(crlayer1+1) && i~=(crlayer2+1)));
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
        case 'conv'
            [backprop, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, ...
                l.opts{:}, cudnn{:}) ;
            res(i).dzdx = updateGradient(res(i).dzdx, backprop);
            clear backprop

      case 'convt'
          [backprop, dzdw{1}, dzdw{2}] = ...
              vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
              res(i+1).dzdx, ...
              'crop', l.crop, 'upsample', l.upsample, ...
              'numGroups', l.numGroups, l.opts{:}, cudnn{:}) ;
            res(i).dzdx = updateGradient(res(i).dzdx, backprop);
            clear backprop


      case 'pool'
        backprop = vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                l.opts{:}, cudnn{:}) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case {'normalize', 'lrn'}
        backprop = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'softmax'
        backprop = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'loss'
        backprop = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'softmaxloss'
        backprop = vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'relu'
        if l.leak > 0, leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          backprop = vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          backprop = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'sigmoid'
        backprop = vl_nnsigmoid(res(i).x, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'noffset'
        backprop = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'spnorm'
        backprop = vl_nnspnorm(res(i).x, l.param, res(i+1).dzdx) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'dropout'
        if testMode
          backprop = res(i+1).dzdx ;
        else
          backprop = vl_nndropout(res(i).x, res(i+1).dzdx, ...
                                     'mask', res(i+1).aux) ;
        end
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'bnorm'
          [backprop, dzdw{1}, dzdw{2}, dzdw{3}] = ...
              vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
              res(i+1).dzdx) ;
          res(i).dzdx = updateGradient(res(i).dzdx, backprop);
          dzdw{3} = dzdw{3} * size(res(i).x,4) ;
          clear backprop
      case 'pdist'
        backprop = vl_nnpdist(res(i).x, l.p, res(i+1).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'bilinearpool'
        backprop = vl_nnbilinearpool(res(i).x, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
      case 'bilinearclpool'
        x1 = res(l.layer1+1).x;
        x2 = res(l.layer2+1).x;
        [y1, y2] = vl_nnbilinearclpool(x1, x2, res(i+1).dzdx);
        res(l.layer1+1).dzdx = updateGradient(res(l.layer1+1).dzdx, y1);
        res(l.layer2+1).dzdx = updateGradient(res(l.layer2+1).dzdx, y2);
      case 'sqrt'
        backprop = vl_nnsqrt(res(i).x, 1e-8, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
      case 'l2norm'
        backprop = vl_nnl2norm(res(i).x, 1e-10, res(i+1).dzdx);
        res(i).dzdx = updateGradient(res(i).dzdx, backprop);
        clear backprop
        
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    
    switch l.type
      case {'conv', 'convt', 'bnorm'}
        if ~opts.accumulate
          res(i).dzdw = dzdw ;
        else
          for j=1:numel(dzdw)
            res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
          end
        end
        dzdw = [] ;
    end
    
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
  if opts.conserveMemory
      res(1).dzdx = [] ;
  end
end


% add up the gradient 
function g = updateGradient(y, backprop)

if isempty(y)
    g = backprop;
else
    g = y + backprop;
end
