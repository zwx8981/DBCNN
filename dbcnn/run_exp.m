dbcnn.name = 'dbcnn' ;
dbcnn.opts = {...
  'type', 'bcnn', ...
  'modela', 'D:\dbcnn\data\models\imagenet-vgg-verydeep-16.mat', ...
  'layera', 30,...
  'modelb', 'D:\dbcnn\data\models\scnn.mat', ...
  'layerb', 18,...
  'shareWeight', false,...
  };


opts.setupNameList = {'dbcnn'};
opts.encoderList = {{dbcnn}}; 
opts.datasetList = {{'live', 1}};  
opts.learningRate = 1e-6;
opts.momentum = 0.9;
opts.batchSize = 8;
opts.numEpoch = 30;
opts.dataset = opts.datasetList{1,1};


srcc = zeros(1,10);
plcc = zeros(1,10);

switch opts.dataset
case 'live'
  subdirec = 'data\checkgpu\live-seed-01';
  modelpath = 'models\LIVE_models';
  datapath = 'data\checkgpu\live-seed-01';
case 'csiq'
  subdirec = 'data\checkgpu\csiq-seed-01';
  modelpath = 'models\CSIQ_models';
  datapath = 'data\checkgpu\csiq-seed-01';
case 'tid'
  subdirec = 'data\checkgpu\tid-seed-01';
  modelpath = 'models\TID_models';
  datapath = 'data\checkgpu\tid-seed-01';
case 'mlive'
  subdirec = 'data\checkgpu\mlive-seed-01';
  modelpath = 'models\MLIVE_models';
  datapath = 'data\checkgpu\mlive-seed-01';
case 'clive'
  subdirec = 'data\checkgpu\clive-seed-01';
  modelpath = 'models\Challen_models';
  datapath = 'data\checkgpu\clive-seed-01';
  
end




for split = 1:10
    mkdir(subdirec);
    rmdir(subdirec,'s');
    imdbpath = fullfile(subdirec,'imdb');
    mkdir(imdbpath);
    copyfile(fullfile(modelpath,num2str(split),'imdb-seed-1.mat'),...
        fullfile(imdbpath,'imdb-seed-1.mat'));
    [options, imdb] = run_experiments_bcnn_train(opts);
    deploy_each_dagnet(opts.numEpoch, datapath);
    options.ftpath = fullfile(subdirec,'fine-tuned-model');
    [srcc(1,split),plcc(1,split),index] = find_bestmodel(opts.numEpoch, imdb, options);
end
