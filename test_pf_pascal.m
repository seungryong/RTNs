%% Script for testing RTNs
%% Written by Seungryong Kim, Yonsei University, Seoul, Korea
clc; clear; close all;
run('vlfeat-0.9.20\toolbox\vl_setup');
run('matconvnet-1.0-beta25\matlab\vl_setupnn.m');
addpath('init_model');
addpath('model');
addpath('function');
addpath('flow-code-matlab');

load('data/RTNs/net-epoch.mat');

net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
net.move('gpu');
net.vars(net.getVarIndex('comp_iter4_flow_transform_x1')).precious = 1;   

%% PF-PASCAL Database
imdbPath = fullfile('data', 'PF-PASCAL.mat');
pf_pascal = load(imdbPath);
pf_pascal = pf_pascal.data;
save_experiments_folder = 'results_pf_pascal'; 

count_data = 0;

for ii = 1:length(pf_pascal.images)
    if pf_pascal.set(ii) == 3
        count_data = count_data+1;
        images_A = pf_pascal.images{ii,1};
        images_B = pf_pascal.images{ii,2};
        
        init =  gpuArray(zeros(16,16,6,'single'));
        init(:,:,1) = 1; init(:,:,5) = 1;

        img1 = gpuArray(single(imresize(images_A,[256,256])));
        img2 = gpuArray(single(imresize(images_B,[256,256])));
        avim = gpuArray(single(cat(3,123.680,116.779,103.939)));
        img1 = bsxfun(@minus,img1,avim);
        img2 = bsxfun(@minus,img2,avim);        
        inputs = {'init', init, 'f1_input', img1, 'f2_input', img2};   
        
        net.eval(inputs);
        results = gather(net.vars(net.getVarIndex('comp_iter4_flow_transform_x1')).value);
        
        vx = imresize(results(:,:,2,:),[224,224],'bilinear');
        vy = imresize(results(:,:,1,:),[224,224],'bilinear');
        
        vx = vx*112;
        vy = vy*112;
        flow = cat(3,vx,vy);
        
        imgWarping = warpImage(im2double(images_B),vx,vy);
        figure(1); imshow(uint8([images_A,imgWarping*256,images_B]));
        figure(2); imshow(flowToColor(flow));
        
        mkdir(fullfile(save_experiments_folder,sprintf('%d',count_data)));

        image1_dir = fullfile(fullfile(save_experiments_folder,sprintf('%d',count_data)),'image1.png');
        image2_dir = fullfile(fullfile(save_experiments_folder,sprintf('%d',count_data)),'image2.png');
        imgWarping_dir = fullfile(fullfile(save_experiments_folder,sprintf('%d',count_data)),'imgWarping.png');
        flow_dir = fullfile(fullfile(save_experiments_folder,sprintf('%d',count_data)),'flow.png');
        flowmat_dir = fullfile(fullfile(save_experiments_folder,sprintf('%d',count_data)),'flow.mat');
        
        imwrite(uint8(images_A*255), image1_dir);
        imwrite(uint8(images_B*255), image2_dir);
        imwrite(uint8(imgWarping*255), imgWarping_dir); 
        imwrite(uint8(flowToColor(flow)), flow_dir);
        save(flowmat_dir,'flow');
    end
end
