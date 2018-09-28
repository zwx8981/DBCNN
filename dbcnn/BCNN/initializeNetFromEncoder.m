
function net = initializeNetFromEncoder(encoder)
% -------------------------------------------------------------------------


if isfield(encoder, 'netb')
    net = buildDagNN(encoder);
else
    if isfield(encoder, 'layera')
        net = addBilinearSimpleNN(encoder);
    else
        if isa(encoder.neta, 'dagnn.DagNN')
            idx = find(arrayfun(@(x) isa(x.block, 'BilinearClPooling'), encoder.neta.layers), 1);
            assert(~isempty(idx), 'no bilinear layer in the network')
            net = encoder.neta;
        else
            idx = find(cellfun(@(x) sum(strcmp(x.type, {'bilinearpool', 'bilinearclpool'})), encoder.neta.layers), 1);
            assert(~isempty(idx), 'no bilinear layer in the network')
            net = encoder.neta;
        end
    end
end

if isa(net, 'dagnn.DagNN')
    net.mode = 'test';
    net.vars(net.getVarIndex('l_1')).precious = 1;
end

%{
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
%}


function net = buildDagNN(encoder)


% transform network to dagnn
% ------------------------------------------------------------------------------------
net = dagnn.DagNN();
net = net.fromSimpleNN(encoder.neta, 'CanonicalNames', true);
meta.meta1 = encoder.neta.meta;

netb = dagnn.DagNN();
netb = netb.fromSimpleNN(encoder.netb, 'CanonicalNames', true);
meta.meta2 = encoder.netb.meta;

net.meta = meta;


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

% % Rename classes
% net.meta.meta1.classes.name = imdb.classes.name;
% net.meta.meta1.classes.description = imdb.classes.name;
% net.meta.meta2.classes.name = imdb.classes.name;
% net.meta.meta2.classes.description = imdb.classes.name;

% add give border for translation data jittering
% if(~strcmp(opts.dataAugmentation{1}, 'f2') && ~strcmp(opts.dataAugmentation{1}, 'none'))
%     net.meta.meta1.normalization.border = 256 - net.meta.meta1.normalization.imageSize(1:2) ;
%     net.meta.meta2.normalization.border = 256 - net.meta.meta2.normalization.imageSize(1:2) ;
% end


function net = addBilinearSimpleNN(encoder)
% -------------------------------------------------------------------------


%{

% network setting
net = vl_simplenn_tidy(net) ;
for l=numel(net.layers):-1:1
    if strcmp(net.layers{l}.type, 'conv')
        net.layers{l}.opts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit};
    end
end
%}


net = encoder.neta;

% stack bilinearpool layer
if isfield(encoder, 'layera') && ~isfield(encoder, 'layerb')
    net.layers{end+1} = struct('type', 'bilinearpool', 'name', 'blp');
else
    assert(isfield(encoder, 'layera') && isfield(encoder, 'layera'), 'Specify both layera and layerb for cross layer bcnn with shared parameters')
    net.layers{end+1} = struct('type', 'bilinearclpool', 'layer1', encoder.layera, 'layer2', encoder.layerb, 'name', 'blcp');
end

% stack normalization
net.layers{end+1} = struct('type', 'sqrt', 'name', 'sqrt_norm');
net.layers{end+1} = struct('type', 'l2norm', 'name', 'l2_norm');


% % Rename classes
% net.meta.classes.name = imdb.classes.name;
% net.meta.classes.description = imdb.classes.name;

% add border for translation data jittering
% if(~strcmp(opts.dataAugmentation{1}, 'f2') && ~strcmp(opts.dataAugmentation{1}, 'none'))
%     net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
% end

 
