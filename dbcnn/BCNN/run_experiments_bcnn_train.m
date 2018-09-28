function [options, imdb] = run_experiments_bcnn_train(opts)

if(~exist('data', 'dir'))
    mkdir('data');
end

  for ii = 1 : numel(opts.datasetList)
    dataset = opts.datasetList{ii} ;
    if iscell(dataset)
      numSplits = dataset{2} ;
      dataset = dataset{1} ;
    else
      numSplits = 1 ;
    end
    for jj = 1 : numSplits
      for ee = 1: numel(opts.encoderList)
        
          [options, imdb] = model_setup('dataset', dataset, ...
			  'encoders', opts.encoderList{ee}, ...
			  'prefix', 'checkgpu', ...  % output folder name
			  'batchSize',opts.batchSize, ...
			  'imgScale', 2, ...       % specify the scale of input images
			  'bcnnLRinit', true, ...   % do logistic regression to initilize softmax layer
			  'dataAugmentation', {'f2','none','none'},...      % do data augmentation [train, val, test]. Only support flipping for train set on current release.
			  'useGpu', 2, ...          %specify the GPU to use. 0 for using CPU
              'learningRate', opts.learningRate, ...
			  'numEpochs', opts.numEpoch, ...
			  'momentum', opts.momentum, ...
			  'keepAspect', true, ...
			  'printDatasetInfo', false, ...
			  'fromScratch', false, ...
			  'rgbJitter', false, ...
			  'useVal', false,...
              'numSubBatches', 1);
          imdb.images.set(imdb.images.set==3) = 2;
          imdb_bcnn_train_dag(imdb, options);
      end
    end
  end
%     bestEpoch = findBestEpoch('D:\zwx_Project\data\checkgpu\chan-seed-01', 'priorityMetric','objective','prune',true);
%     path = fullfile('D:\zwx_Project\data\checkgpu\chan-seed-01',strcat('net-epoch-',num2str(bestEpoch),'.mat'));
%     net = load(path); 
% %     net = net.net;
%     net = net_deploy(net);
%     save('D:\zwx_Project\data\checkgpu\chan-seed-01\net-deployed.mat', 'net',  '-v7.3') ;
end

%{
The following are the setting we run in which fine-tuning works stable without GPU memory issues on Nvidia K40.
m-m model: batchSize 64, momentum 0.9
d-m model: batchSize 1, momentum 0.3
%}

