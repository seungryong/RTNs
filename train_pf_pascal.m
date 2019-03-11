%% Script for learning RTNs
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
function train_pf_pascal(varargin)
    run('vlfeat-0.9.20\toolbox\vl_setup');
    run('matconvnet-1.0-beta25\matlab\vl_setupnn.m');
    addpath('init_model');
    addpath('model');
    addpath('function');
    
    load('data/imdb_pf_pascal.mat');
    
    init_model = true; % Using pretrained model as an initial parameter (or not)
    net = init_RTNs(init_model);

    trainOpts.batchSize = 16;
    trainOpts.numEpochs = 20;
    trainOpts.continue = true;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 1e-5;
    trainOpts.derOutputs = {'objective1', 1};
    trainOpts.expDir = 'data/RTNs';
    trainOpts = vl_argparse(trainOpts, varargin);
    
    cnn_train_dag(net, imdb, getBatch, trainOpts);    
end

function inputs = getBatch()
    inputs = @(imdb,batch) getBatch_pf_pascal(imdb,batch);
end
