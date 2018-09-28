function net = initializeNetworkTwoStreams(imdb, encoderOpts, opts)

% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved.
%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

% This code is used for initializing asymmetric B-CNN network 

% -------------------------------------------------------------------------
scal = 1 ;
init_bias = 0.1;
% numClass = length(imdb.classes.name);

assert(~isempty(encoderOpts.modela) && ~isempty(encoderOpts.modelb), 'Error: at least one of the network is not specified')

% load the pre-trained models
encoder.neta = load(encoderOpts.modela);
encoder.neta.layers = encoder.neta.layers(1:encoderOpts.layera);
encoder.netb = load(encoderOpts.modelb);
encoder.netb.layers = encoder.netb.layers(1:encoderOpts.layerb);
encoder.regionBorder = 0.05;
encoder.type = 'bcnn';
encoder.normalization = 'sqrt_L2';

% move models to GPU
if ~isempty(opts.useGpu)
    encoder.neta = vl_simplenn_move(encoder.neta, 'gpu') ;
    encoder.netb = vl_simplenn_move(encoder.netb, 'gpu') ;
else
    encoder.neta = vl_simplenn_move(encoder.neta, 'cpu') ;
    encoder.netb = vl_simplenn_move(encoder.netb, 'cpu') ;
end
for l=numel(encoder.neta.layers):-1:1
    if strcmp(encoder.neta.layers{l}.type, 'conv')
        encoder.neta.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    end
end
for l=numel(encoder.netb.layers):-1:1
    if strcmp(encoder.netb.layers{l}.type, 'conv')
        encoder.netb.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    end
end

% get the feature dimension for both layers
netaInfo = vl_simplenn_display(encoder.neta);
netbInfo = vl_simplenn_display(encoder.netb);
ch1 = netaInfo.dataSize(3, encoderOpts.layera+1);
ch2 = netbInfo.dataSize(3, encoderOpts.layerb+1);
dim = ch1*ch2;


% add batch normalization layers
% ------------------------------------------------------------------------------------
if opts.batchNormalization
    for l=numel(encoder.neta.layers):-1:1
        if isfield(encoder.neta.layers{l}, 'weights') 
            if isempty(encoder.neta.layers{l}.weights)
                continue;
            end
            ndim = size(encoder.neta.layers{l}.weights{1}, 4);
            
            
            layer = struct('type', 'bnorm', 'name', sprintf('bna%s',num2str(l)), ...
                'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), [zeros(ndim, 1, 'single'), ones(ndim, 1, 'single')]}}, ...
                'learningRate', [2 1 0.05], ...
                'weightDecay', [0 0]) ;
            
            encoder.neta.layers = horzcat(encoder.neta.layers(1:l), layer, encoder.neta.layers(l+1:end)) ;
            
        end
        encoder.neta = simpleRemoveLayersOfType(encoder.neta,'lrn');
    end
    
    
    for l=numel(encoder.netb.layers):-1:1
        if isfield(encoder.netb.layers{l}, 'weights') 
            if isempty(encoder.netb.layers{l}.weights)
                continue;
            end
            ndim = size(encoder.netb.layers{l}.weights{1}, 4);
            
            layer = struct('type', 'bnorm', 'name', sprintf('bnb%s',num2str(l)), ...
                'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single'), [zeros(ndim, 1, 'single'), ones(ndim, 1, 'single')]}}, ...
                'learningRate', [2 1 0.05], ...
                'weightDecay', [0 0]) ;
            
            encoder.netb.layers = horzcat(encoder.netb.layers(1:l), layer, encoder.netb.layers(l+1:end)) ;
            
        end
    end
    
    encoder.netb = simpleRemoveLayersOfType(encoder.netb,'lrn');
end
% ------------------------------------------------------------------------------------



% transform network to dagnn
% ------------------------------------------------------------------------------------
net = dagnn.DagNN();

for i = 1:size(encoder.neta.layers,2)
    encoder.neta.layers{i}.name = strrep(encoder.neta.layers{i}.name,'-','_');
end

net = net.fromSimpleNN(encoder.neta, 'CanonicalNames', true);
meta.meta1 = encoder.neta.meta;

netb = dagnn.DagNN();

for i = 1:size(encoder.netb.layers,2)
    encoder.netb.layers{i}.name = strrep(encoder.netb.layers{i}.name,'-','_');
end

netb = netb.fromSimpleNN(encoder.netb, 'CanonicalNames', true);
meta.meta2 = encoder.netb.meta;

net.meta = meta;

net.meta.meta1.normalization.keepAspect = opts.keepAspect;
net.meta.meta2.normalization.keepAspect = opts.keepAspect;

for i=1:numel(netb.layers)
    layerName = strcat('netb_', netb.layers(i).name);
    input = strcat('netb_', netb.layers(i).inputs);
    output = strcat('netb_', netb.layers(i).outputs);
    params = strcat('netb_', netb.layers(i).params);
    %         net.layers(end+1) = netb.layers(i);
    net.addLayer(layerName, netb.layers(i).block, input, output, params);
    
    for f = 1:numel(params)
        varId = net.getParamIndex(params{f});
        varIdb = netb.getParamIndex(netb.layers(i).params{f});
        if strcmp(net.device, 'gpu')
            net.params(varId).value = gpuArray(netb.params(varIdb).value);
        else
            net.params(varId).value = netb.params(varIdb).value;
        end
    end
end

clear netb
% ------------------------------------------------------------------------------------


% Add bilinearpool layer
bp_layer = {encoder.neta.layers{end}.name, strcat('netb_', encoder.netb.layers{end}.name)};
inputLayerIndex = net.getLayerIndex(bp_layer);
in1 = net.layers(inputLayerIndex(1)).outputs;
assert(length(in1) == 1);
in2 = net.layers(inputLayerIndex(2)).outputs;
assert(length(in2) == 1);
input = cat(2, in1, in2);
layerName = 'bilr_1';
output = 'b_1';
net.addLayer(layerName, BilinearClPooling('normalizeGradients', false), ...
    input, output);

% Square-root layer
layerName = sprintf('sqrt_1');
input = output;
output = 's_1';
net.addLayer(layerName, SquareRoot(), {input}, output);


% L2 normalization layer
layerName = 'l2_1';
input = output;
bpoutput = 'l_1';
net.addLayer(layerName, L2Norm(), {input}, bpoutput);


% ------------------------------------------------------------------------------------


% build a linear classifier netc

% initialW = 0.001/scal *randn(1,1,dim, numClass,'single');
% initialBias = init_bias.*ones(1, numClass, 'single');
% netc.layers = {};
% netc.layers{end+1} = struct('type', 'conv', 'name', 'classifier', ...
%     'weights', {{initialW, initialBias}}, ...
%     'stride', 1, ...
%     'pad', 0, ...
%     'learningRate', [1000 1000], ...
%     'weightDecay', [0 0]) ;
% netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

netc.layers = {};
netc = add_block(netc, opts, 'fc', 1, 1, dim, 1, 1, 0) ;
netc.layers(end) = [] ;
netc.layers{end}.weights{1} = netc.layers{end}.weights{1} / 10 ;
netc.layers{end+1} = struct('type', 'pdist', 'name', 'loss','noRoot',true,'aggregate',true,'p',2,'Epsilon',1e-6) ;

netc = vl_simplenn_tidy(netc) ;


% pretrain the linear classifier with logistic regression
if(opts.bcnnLRinit && ~opts.fromScratch)
    if exist(fullfile(opts.expDir, 'initial_fc.mat'))
        load(fullfile(opts.expDir, 'initial_fc.mat'), 'netc') ;
        
    else
        train = find(ismember(imdb.images.set, [1 2]));
        % compute and cache the bilinear cnn features
        if ~exist(opts.nonftbcnnDir)
            mkdir(opts.nonftbcnnDir)
            
            batchSize = 8;
            
            net.meta.meta1.normalization.border = [0 0];
			net.meta.meta1.normalization.interpolation = 'bicubic';
%             net.meta.meta1.normalization = rmfield(net.meta.meta1.normalization,'imageStd');
%             net.meta.meta1.normalization.cropSize = [0.448 0.448];
            bopts(1) = net.meta.meta1.normalization;
            net.meta.meta2.normalization.interpolation = 'bicubic';
            net.meta.meta2.normalization.averageImage = reshape(net.meta.meta2.normalization.averageImage,[1,1,3]);
%             net.meta.meta2.normalization.border = [32 32];
            net.meta.meta2.normalization.border = [0 0];
%             net.meta.meta2.normalization = rmfield(net.meta.meta2.normalization,'cropSize');
            bopts(2) = net.meta.meta2.normalization;
            bopts(1).numThreads = opts.numFetchThreads ;
            bopts(2).numThreads = opts.numFetchThreads ;
            
            for i=1:numel(bopts)
                bopts(i).transformation = 'none' ;
                bopts(i).rgbVariance = [];
                bopts(i).scale = opts.imgScale ;
                bopts(i).dataset = opts.dataset;
            end
            
            useGpu = numel(opts.train.gpus) > 0 ;
            if useGpu
                net.move('gpu') ;
            end
            
            getBatchFn = getBatchDagNNWrapper(bopts, useGpu) ;
            
            for t=1:batchSize:numel(train)
                fprintf('Initialization: extracting bcnn feature of batch %d/%d\n', ceil(t/batchSize), ceil(numel(train)/batchSize));
                batch = train(t:min(numel(train), t+batchSize-1));
                input = getBatchFn(imdb, batch) ;
                if opts.train.prefetch
                    nextBatch = train(t+batchSize:min(t+2*batchSize-1, numel(train))) ;
                    getBatchFn(imdb, nextBatch) ;
                end
                
                input = input(1:4);
                net.mode = 'test' ;
                net.eval(input);
                fIdx = net.getVarIndex('l_1');
                code_b = net.vars(fIdx).value;
                code_b = squeeze(gather(code_b));
                
                for i=1:numel(batch)
                    code = code_b(:,i);
                    savefast(fullfile(opts.nonftbcnnDir, ['bcnn_nonft_', num2str(batch(i), '%05d')]), 'code');
                end
            end
            
            % move back to cpu
            if useGpu
                net.move('cpu') ;
            end
        end
        
        
        bcnndb = imdb;
        tempStr = sprintf('%05d\t', train);
        tempStr = textscan(tempStr, '%s', 'delimiter', '\t');
        bcnndb.images.name = strcat('bcnn_nonft_', tempStr{1}');
        bcnndb.images.id = bcnndb.images.id(train);
        bcnndb.images.label = bcnndb.images.label(train);
        bcnndb.images.set = bcnndb.images.set(train);
        bcnndb.imageDir = opts.nonftbcnnDir;
        
        %train logistic regression
        [netc, info] = cnn_train(netc, bcnndb, @getBatch_bcnn_fromdisk, opts.inittrain, ...
            'conserveMemory', true, 'numEpochs',50,'errorFunction','none');
        
        save(fullfile(opts.expDir, 'initial_fc.mat'), 'netc', '-v7.3') ;
    end
end

% Dropout layer
% layerName = 'dropout';
% input = bpoutput;
% dropout = 'dropout';
% net.addLayer(layerName, dagnn.DropOut('rate', 0.7), {input}, dropout);

% Initial the classifier to the pretrained weights
layerName = 'regressor';
param(1).name = 'convclass_f';
param(1).value = netc.layers{1}.weights{1};
param(2).name = 'convattr_b';
param(2).value = netc.layers{1}.weights{2};
net.addLayer(layerName, dagnn.Conv(), {bpoutput}, 'score', {param(1).name param(2).name});
% net.addLayer(layerName, dagnn.Conv(), {dropout}, 'score', {param(1).name param(2).name});

for f = 1:2,
    varId = net.getParamIndex(param(f).name);
    if strcmp(net.device, 'gpu')
        net.params(varId).value = gpuArray(param(f).value);
    else
        net.params(varId).value = param(f).value;
    end
    net.params(varId).learningRate = 1000;
    net.params(varId).weightDecay = 0;
end

%higher lr for VGG-16
% ParamIdx = net.getParamIndex([net.layers(31:48).params]);
% Rate = 0.1;
% [net.params(ParamIdx).learningRate] = deal(Rate);

% add loss functions
net.addLayer('loss', dagnn.PDist('loss', 'pdist','p',2,'noRoot',true), {'score','label'}, 'objective');
% net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'score','label'}, 'objective');
% net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'score','label'}, 'top1error');
% net.addLayer('top5e', dagnn.Loss('loss', 'topkerror'), {'score','label'}, 'top5error');
clear netc

net.mode = 'normal';


if(opts.fromScratch)
    for i=1:numel(net.layers)
        %if isempty(net.layers{i}.paramIndexes), continue ; end
        if ~isa(net.layers(i).block, 'dagnn.Conv'), continue ; end
        for j=1:numel(net.layers(i).params)
            paramIdx = net.getParamIndex(net.layers(i).params{j});
            if ndims(net.params(paramIdx).value) == 2
                net.params(paramIdx).value = init_bias*ones(size(net.params(paramIdx).value), 'single');
            else
                net.params(paramIdx).value = 0.01/scal * randn(net.params(paramIdx).value, 'single');
            end
        end
    end
end


% Rename classes
% net.meta.meta1.classes.name = imdb.classes.name;
% net.meta.meta1.classes.description = imdb.classes.name;
% net.meta.meta2.classes.name = imdb.classes.name;
% net.meta.meta2.classes.description = imdb.classes.name;

% add give border for translation data jittering
if(~strcmp(opts.dataAugmentation{1}, 'f2') && ~strcmp(opts.dataAugmentation{1}, 'none'))
    net.meta.meta1.normalization.border = 256 - net.meta.meta1.normalization.imageSize(1:2) ;
    net.meta.meta2.normalization.border = 256 - net.meta.meta2.normalization.imageSize(1:2) ;
end




function [im,labels] = getBatch_bcnn_fromdisk(imdb, batch)
% -------------------------------------------------------------------------

im = cell(1, numel(batch));
for i=1:numel(batch)
    load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
    im{i} = code;
end
im = cat(2, im{:});
im = reshape(im, 1, 1, size(im,1), size(im, 2));
labels = imdb.images.label(batch) ;




function layers = simpleFindLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;

% -------------------------------------------------------------------------
function net = simpleRemoveLayersOfType(net, type)
% -------------------------------------------------------------------------
layers = simpleFindLayersOfType(net, type) ;
net.layers(layers) = [] ;



% --------------------------------------------------------------------
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
% --------------------------------------------------------------------
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                           'weights', {{init_weight(opts, h, w, in, out, 'single'), ...
                             ones(out, 1, 'single')*opts.initBias}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', 1, ...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0], ...
                           'opts', {convOpts}) ;
if opts.batchNormalization
  net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%s',id), ...
                             'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single'), ...
                               zeros(out, 2, 'single')}}, ...
                             'epsilon', 1e-4, ...
                             'learningRate', [2 1 0.1], ...
                             'weightDecay', [0 0]) ;
end
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end
