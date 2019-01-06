function [srcc, plcc, index] = find_bestmodel(epoch, imdb, options)


% imdb = load('D:\zwx_Project\zwx_IQA\Chan_models\whole\imdb-seed-1.mat');
% imdb = imdb.imdb;
% net = load('zwx_IQA\Chan_models\1\net-deployed.mat');
srocc = zeros(1,epoch);
plc = zeros(1,epoch);
for k = 1:epoch
    name = strcat('net-deployed-',num2str(k),'.mat');
    dagnet = load(fullfile(options.ftpath,name));    
    dagnet = dagnn.DagNN.loadobj(dagnet.net) ;

    move(dagnet, 'gpu')
    dagnet.mode = 'test';
    dagnet.conserveMemory = 0;
    
    opts.numThreads = 8;
    averageImage = reshape(dagnet.meta.meta1.normalization.averageImage,[1,1,3]);
    set = imdb.images.set;
    
    sel = find(set(:)==2);
    path = cell(1,size(sel,1));
    label = zeros(1,size(sel,1));
    for i = 1:size(sel)
        path{1,i} = imdb.images.name{1,sel(i)};
        label(i) = imdb.images.label(sel(i));
    end   
    scores = zeros(1,size(sel,1));
    for i = 1:size(sel)
        imagePaths = {fullfile(options.dataset_path,path{i})};
        im = imread(imagePaths{1,1});
        if options.dataset == 'mlive'
            im = imresize(im,0.75);
        end
        im = gpuArray(single(im));
        data = bsxfun(@minus,im,averageImage);
        inputs = {'input', data, 'netb_input', data} ;
        dagnet.eval(inputs) ;
        scores(i) = gather(dagnet.vars(54).value);
    end
    
    [srocc(k),~,plc(k),~] = verify_performance(scores,label);     
end
if max(srocc) < 0
    [srcc,index] = min(srocc);
else
    [srcc,index] = max(srocc);
end
plcc = plc(index);
