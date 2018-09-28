
% BilinearClPooling is the dagnn wapper of vl_nnbilinearclpool which 
% computes outer product of outputs of two layers and pool the features 
% across all locations

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).


classdef Bilinear < dagnn.Filter
  properties
    normalizeGradients = false;
  end

  methods
    function outputs = forward(obj, inputs, params)
        if numel(inputs)==1
            outputs{1} = vl_nnbilinear(inputs{1});
        else
            outputs{1} = vl_nnbilinearcl(inputs{1}, inputs{2});
        end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        if numel(inputs)==1
            derInputs{1} = vl_nnbilinear(inputs{1}, derOutputs{1});
        else
            [derInputs{1}, derInputs{2}] = vl_nnbilinearcl(inputs{1}, inputs{2}, derOutputs{1});
        end
      if obj.normalizeGradients
          for i=1:numel(derInputs)
              gradNorm = sum(abs(derInputs{i}(:))) + 1e-8;
              derInputs{i} = derInputs{i}/gradNorm;
          end
      end
      derParams = {} ;
    end
    
    
    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Bilinear(varargin)
      obj.load(varargin) ;
    end
  end
end

