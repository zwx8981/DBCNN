
% BilinearClPooling is the dagnn wapper of vl_nnbilinearclpool which 
% computes outer product of outputs of two layers and pool the features 
% across all locations

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


classdef BilinearClPooling < dagnn.Filter
  properties
    method = 'sum'
    normalizeGradients = false;
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnbilinearclpool(inputs{1}, inputs{2});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derInputs{2}] = vl_nnbilinearclpool(inputs{1}, inputs{2}, derOutputs{1});
      if obj.normalizeGradients
        gradNorm = sum(abs(derInputs{1}(:))) + 1e-8;
        derInputs{1} = derInputs{1}/gradNorm;
        
        gradNorm = sum(abs(derInputs{2}(:))) + 1e-8;
        derInputs{2} = derInputs{2}/gradNorm;
      end
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = BilinearClPooling(varargin)
      obj.load(varargin) ;
    end
  end
end

