% L2Norm is the dagnn wrapper which normalizes the features to be norm 1 in
% 2-norm at each location

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN. It uses MATCONVNET package and is made 
% available under the terms of the BSD license (see the COPYING file).

classdef L2Norm < dagnn.ElementWise
  properties
    param = 1e-10;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnl2norm(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnl2norm(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function obj = SpatialNorm(varargin)
      obj.load(varargin) ;
    end
  end
end
