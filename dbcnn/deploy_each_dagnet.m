function deploy_each_dagnet(epoch, datapath)

for i = 1:epoch
    fileName = strcat('net-epoch-',num2str(i),'.mat');
    fileName = fullfile(datapath,fileName);
    [net, info] = loadState_deploy(fileName);
    net = net_deploy(net,1) ;
    deployname = strcat('net-deployed-',num2str(i),'.mat');
    save(fullfile(datapath, 'fine-tuned-model', deployname), 'net', 'info', '-v7.3');
end

