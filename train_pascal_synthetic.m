%% Script for learning RTNs with synthetic data
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
function train_pascal_synthetic(varargin)
    run('vlfeat-0.9.20\toolbox\vl_setup');
    run('matconvnet-1.0-beta25\matlab\vl_setupnn.m');
    addpath('init_model');
    addpath('model');
    addpath('function');
    
    load('data/imdb_pascal_synthetic.mat');

    net = init_RTNs_synthetic();

    trainOpts.batchSize = 16;
    trainOpts.numEpochs = 10;
    trainOpts.continue = true;
    trainOpts.gpus = 1;
    trainOpts.learningRate = 1e-5;
    trainOpts.derOutputs = {'objective1', 1};
    trainOpts.expDir = 'data/RTNs_synthetic';
    trainOpts = vl_argparse(trainOpts, varargin);
    
    cnn_train_dag(net, imdb, getBatch, trainOpts);    
end

function inputs = getBatch()
    inputs = @(imdb,batch) getBatch_pascal_synthetic(imdb,batch);
end
