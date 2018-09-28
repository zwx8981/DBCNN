function imo = imdb_get_batch_bcnn(images, varargin)
% imdb_get_batch_bcnn  Load, preprocess, and pack images for BCNN evaluation
% For asymmetric model, the function preprocesses the images twice for two networks
% separately.

% OUTPUT
% imo: a cell array where each element is a cell array of images.
%       For symmetric bcnn model, numel(imo) will be 1 and imo{1} will be a
%       cell array of images
%       For asymmetric bcnn, numel(imo) will be 2. imo{1} is a cell array containing the preprocessed images for network A 
%       Similarly, imo{2} is a cell array containing the preprocessed images for network B

%
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).
%
% This file modified from CNN_IMAGENET_GET_BATCH of MatConvNet


for i=1:numel(varargin{1})
    opts(i).imageSize = [227, 227] ;
    opts(i).border = [0, 0] ;
    opts(i).keepAspect = true;
    opts(i).numAugments = 1 ;
    opts(i).transformation = 'none' ;
    opts(i).averageImage = [] ;
    opts(i).rgbVariance = zeros(0,3,'single') ;
    opts(i).interpolation = 'bilinear' ;
    opts(i).numThreads = 1 ;
    opts(i).prefetch = false;
    opts(i).scale = 1;
    opts(i).cropsize = [0.875,0.875];
    opts(i).dataset = 'live';
    opts(i) = vl_argparse(opts(i), {varargin{1}(i),varargin{2:end}});

    
    if(i==1)
        
        
        if ~isempty(opts(i).rgbVariance) && isempty(opts(i).averageImage)
            opts(i).averageImage = zeros(1,1,3) ;
        end
        if numel(opts(i).averageImage) == 3
            opts(i).averageImage = reshape(opts(i).averageImage, 1,1,3) ;
        end
        if ~isempty(opts(i).rgbVariance)
            rgbjitt = opts(i).rgbVariance * randn(3,numel(images));
        else
            rgbjitt = [];
        end
        
        
        
        % fetch is true if images is a list of filenames (instead of
        % a cell array of images)
        % fetch = numel(images) > 1 && ischar(images{1}) ;
        fetch = ischar(images{1}) ;
        
        % prefetch is used to load images in a separate thread
        prefetch = fetch & opts(i).prefetch ;
        
        
        if prefetch
            vl_imreadjpeg(images, 'numThreads', opts(i).numThreads, 'prefetch') ;
            imo = [] ;
            return ;
        end
        if fetch
            im = vl_imreadjpeg(images,'numThreads', opts(i).numThreads) ;
        else
            im = images ;
        end

    end
    
    % preprocess images for the ith network
%     imo{i} = get_batch_fun(images, im,  opts(i), transformations, tfs, rgbjitt);
    imo{i} = get_batch_fun(images, im,  opts(i));
end



function imo = get_batch_fun(images, im, opts)

opts.border = round(opts.border.*opts.scale);
if(opts.scale ~= 1)
    opts.averageImage = mean(mean(opts.averageImage, 1),2);
end

if strcmp(opts.dataset,'live') || strcmp(opts.dataset,'clive')

    if strcmp(opts.dataset,'live')
        imo = zeros(432, 432, 3, ...
        numel(images)*opts.numAugments, 'single') ;
    else
        imo = zeros(480, 480, 3, ...
        numel(images)*opts.numAugments, 'single') ;
    end

    si=1;
    for i=1:numel(images)
    
      % acquire image
      if isempty(im{i})
        imt = imread(images{i}) ;
        imt = single(imt) ; % faster than im2single (and multiplies by 255)
      else
        imt = im{i} ;
      end
      
      h = size(imt,1);w = size(imt,2);
      
      top = randi(h - 432);
      bottom = h - 432 - top;
      left = randi(w - 432);
      right = w - 432 - left;
      imt = vl_nncrop(imt,[top,bottom,left,right]);
     
      
      offset = opts.averageImage ;
      if ~isempty(opts.averageImage)
          offset = opts.averageImage ;
          offset = reshape(offset,[1,1,3]);
    %       if ~isempty(opts.rgbVariance)
    %           offset = bsxfun(@plus, offset, reshape(rgbjitt(:,i), 1,1,3)) ;
    %       end
          imo(:,:,:,i) = bsxfun(@minus, imt, offset) ; 
      else
          imo(:,:,:,si) = imt ;
      end
    end
elseif strcmp(opts.dataset,'csiq') || strcmp(opts.dataset,'tid')

    if strcmp(opts.dataset,'csiq')
        imo = zeros(512, 512, 3, ...
        numel(images)*opts.numAugments, 'single') ;
    else
        imo = zeros(384, 512, 3, ...
        numel(images)*opts.numAugments, 'single') ; 
    end

    si=1;
    for i=1:numel(images)
    
      % acquire image
      if isempty(im{i})
        imt = imread(images{i}) ;
        imt = single(imt) ; % faster than im2single (and multiplies by 255)
      else
        imt = im{i} ;
      end
      
      imo(:,:,:,i) = imt;
      offset = opts.averageImage ;
      if ~isempty(opts.averageImage)
          offset = opts.averageImage ;
    %       if ~isempty(opts.rgbVariance)
    %           offset = bsxfun(@plus, offset, reshape(rgbjitt(:,i), 1,1,3)) ;
    %       end
          imo(:,:,:,i) = bsxfun(@minus, imt, offset) ; 
      else
          imo(:,:,:,si) = imt ;
      end
    end
else
    opts.border = round(opts.border.*opts.scale);
    if(opts.scale ~= 1)
        opts.averageImage = mean(mean(opts.averageImage, 1),2);
    end
    
    
    imo = zeros(540, 960, 3, ...
                numel(images)*opts.numAugments, 'single');      
    si=1;
    for i=1:numel(images)
    
      % acquire image
      if isempty(im{i})
        imt = imread(images{i}) ;
        imt = single(imt) ; % faster than im2single (and multiplies by 255)
      else
        imt = im{i} ;
      end
      
      imt = imresize(imt, 0.75); %% GPU memory
      
      imo(:,:,:,i) = imt;
      offset = opts.averageImage ;
      if ~isempty(opts.averageImage)
          offset = opts.averageImage ;
    %       if ~isempty(opts.rgbVariance)
    %           offset = bsxfun(@plus, offset, reshape(rgbjitt(:,i), 1,1,3)) ;
    %       end
          imo(:,:,:,i) = bsxfun(@minus, imt, offset) ; 
      else
          imo(:,:,:,si) = imt ;
      end
    end
end

