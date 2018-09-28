function [net, stats] = loadState_deploy(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;