img_path = 'your image path';
net_path = 'dbcnn model path';

mode = 'gpu'; %or cpu

dagnet = load(net_path);   
dagnet = dagnn.DagNN.loadobj(dagnet.net) ;
move(dagnet, mode)
dagnet.mode = 'test';
dagnet.conserveMemory = 0;
    
opts.numThreads = 8;
averageImage = reshape(dagnet.meta.meta1.normalization.averageImage,[1,1,3]);
    
im = imread(img_path);

if strcmp(mode,'gpu') == 1
    im = gpuArray(single(im));
end

data = bsxfun(@minus,im,averageImage);
inputs = {'input', data, 'netb_input', data} ;
dagnet.eval(inputs) ;

%predicting the objective quality score of the input image
predict_score = gather(dagnet.vars(54).value);
