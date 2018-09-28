% SquareRoot is the dagnn wrapper which performs square root normalization 
% for the input features at each location

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN. It uses MATCONVNET package and is made 
% available under the terms of the BSD license (see the COPYING file).

classdef SquareRoot < dagnn.ElementWise
  properties
    param = 1e-8;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnsqrt(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnsqrt(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = SquareRoot(varargin)
      obj.load(varargin) ;
    end
  end
end
