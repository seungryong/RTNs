function net = init_RTNs(init_model)
    maxdisp = 9;

    %% VGG-Net Initialization
    % -------------------------------------------------------------------------
    %                    Load & download the source model if needed (VGG VD 16)
    % -------------------------------------------------------------------------
    sourceModelPath = 'model/imagenet-vgg-verydeep-19.mat';
    if ~exist(sourceModelPath)
      fprintf('downloading VGG model\n') ;
      mkdir(fileparts(sourceModelPath)) ;
      urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', sourceModelPath) ;
    end
    net_vgg = load(sourceModelPath);
    
    net_vgg_first.layers = {};
    for ii = 1:30
        net_vgg_first.layers{end+1} = net_vgg.layers{ii};
    end    
    net_vgg_first = vl_simplenn_tidy(net_vgg_first);
    net_vgg_first = dagnn.DagNN.fromSimpleNN(net_vgg_first, 'canonicalNames', true); 
    net_vgg_first.addLayer('normalization_vgg_first1', L2norm_Layer(), {'x21'}, {'x21_norm'});
    net_vgg_first.addLayer('normalization_vgg_first2', L2norm_Layer(), {'x30'}, {'x30_norm'});
    for i = 1:26
        net_vgg_first.params(i).learningRate = 0.1;
    end
    
    net_vgg_last.layers = {};
    for ii = 20:21
        net_vgg_last.layers{end+1} = net_vgg.layers{ii};
    end  
    net_vgg_last = vl_simplenn_tidy(net_vgg_last);
    net_vgg_last = dagnn.DagNN.fromSimpleNN(net_vgg_last, 'canonicalNames', true);   
    net_vgg_last.layers(1).block.stride = [3,3];
    net_vgg_last.layers(1).block.pad = [0,0,0,0];
    net_vgg_last.addLayer('normalization_vgg_last', L2norm_Layer(), {'x2'}, {'norm'});
    for i = 1:2
        net_vgg_last.params(i).learningRate = 0.1;
    end
    
    net_vgg_last1.layers = {};
    for ii = 29:30
        net_vgg_last1.layers{end+1} = net_vgg.layers{ii};
    end  
    net_vgg_last1 = vl_simplenn_tidy(net_vgg_last1);
    net_vgg_last1 = dagnn.DagNN.fromSimpleNN(net_vgg_last1, 'canonicalNames', true);   
    net_vgg_last1.layers(1).block.stride = [3,3];
    net_vgg_last1.layers(1).block.pad = [0,0,0,0];
    net_vgg_last1.addLayer('normalization_vgg_last1', L2norm_Layer(), {'x2'}, {'norm'});
    for i = 1:2
        net_vgg_last1.params(i).learningRate = 0.1;
    end
    
    netStruct = net_vgg_first.saveobj;
    scr_netStruct = netNamePrefix(netStruct,'f1_','f1_','');
    tar_netStruct = netNamePrefix(netStruct,'f2_','f2_','');
    netStructFused = fuseNetStruct(scr_netStruct,tar_netStruct);
    net = dagnn.DagNN.loadobj(netStructFused);
    
    %% Geometric Transformation Estimation
    % Conv1
    geo_net = dagnn.DagNN();
    geo_conv1 = dagnn.Conv('size', [9 9 81 64], 'hasBias', true, 'pad', 4, 'stride', 1, 'dilate', 1);
    geo_net.addLayer('geo_conv1', geo_conv1, {'corr_vol_norm'}, {'x1'}, {'geo_conv1_f','geo_conv1_b'});
    geo_net.addLayer('geo_relu1', dagnn.ReLU(), {'x1'}, {'x2'});
    
    geo_conv1b = dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv1b', geo_conv1b, {'x2'}, {'x3'}, {'geo_conv1b_f','geo_conv1b_b'});
    geo_net.addLayer('geo_relu1b', dagnn.ReLU(), {'x3'}, {'x4'});

    geo_net.addLayer('geo_pooling1', dagnn.Pooling('poolSize',[2,2],'pad',[0,1,0,1],'stride',[2,2],'method','max'), {'x4'},{'x5'});

    % Conv2
    geo_conv2 = dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv2', geo_conv2, {'x5'}, {'x6'}, {'geo_conv2_f','geo_conv2_b'});
    geo_net.addLayer('geo_relu2', dagnn.ReLU(), {'x6'}, {'x7'});

    geo_conv2b = dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv2b', geo_conv2b, {'x7'}, {'x8'}, {'geo_conv2b_f','geo_conv2b_b'});
    geo_net.addLayer('geo_relu2b', dagnn.ReLU(), {'x8'}, {'x9'});

    geo_net.addLayer('geo_pooling2', dagnn.Pooling('poolSize',[2,2],'pad',[0,1,0,1],'stride',[2,2],'method','max'), {'x9'},{'x10'});

    % Conv3
    geo_conv3 = dagnn.Conv('size', [3 3 128 256], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv3', geo_conv3, {'x10'}, {'x11'}, {'geo_conv3_f','geo_conv3_b'});
    geo_net.addLayer('geo_relu3', dagnn.ReLU(), {'x11'}, {'x12'});

    geo_conv3b = dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv3b', geo_conv3b, {'x12'}, {'x13'}, {'geo_conv3b_f','geo_conv3b_b'});
    geo_net.addLayer('geo_relu3b', dagnn.ReLU(), {'x13'}, {'x14'});
    
    geo_net.addLayer('geo_pooling3', dagnn.Pooling('poolSize',[2,2],'pad',[0,1,0,1],'stride',[2,2],'method','max'), {'x14'},{'x15'});
    
    % Conv4
    geo_conv4 = dagnn.Conv('size', [3 3 256 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv4', geo_conv4, {'x15'}, {'x16'}, {'geo_conv4_f','geo_conv4_b'});
    geo_net.addLayer('geo_relu4', dagnn.ReLU(), {'x16'}, {'x17'});

    geo_conv4b = dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv4b', geo_conv4b, {'x17'}, {'x18'}, {'geo_conv4b_f','geo_conv4b_b'});
    geo_net.addLayer('geo_relu4b', dagnn.ReLU(), {'x18'}, {'x19'});
    
    geo_net.addLayer('geo_pooling4', dagnn.Pooling('poolSize',[2,2],'pad',[0,1,0,1],'stride',[2,2],'method','max'), {'x19'},{'x20'});
    
    % Conv5
    geo_conv5 = dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv5', geo_conv5, {'x20'}, {'x21'}, {'geo_conv5_f','geo_conv5_b'});
    geo_net.addLayer('geo_relu5', dagnn.ReLU(), {'x21'}, {'x22'});

    geo_conv5b = dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv5b', geo_conv5b, {'x22'}, {'x23'}, {'geo_conv5b_f','geo_conv5b_b'});
    geo_net.addLayer('geo_relu5b', dagnn.ReLU(), {'x23'}, {'x24'});
    
    filters = single(bilinear_u(4,1,512));
    geo_net.addLayer('geo_deconv1', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'x24', 'x25', 'geo_deconv1_f');
    
    % Conv6
    geo_net.addLayer('geo_concat1', dagnn.Concat('dim',3), {'x25', 'x19'}, {'x26'});

    geo_conv6 = dagnn.Conv('size', [3 3 512+512 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv6', geo_conv6, {'x26'}, {'x27'}, {'geo_conv6_f','geo_conv6_b'});
    geo_net.addLayer('geo_relu6', dagnn.ReLU(), {'x27'}, {'x28'});

    geo_conv6b = dagnn.Conv('size', [3 3 512 512], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv6b', geo_conv6b, {'x28'}, {'x29'}, {'geo_conv6b_f','geo_conv6b_b'});
    geo_net.addLayer('geo_relu6b', dagnn.ReLU(), {'x29'}, {'x30'});    
    
    filters = single(bilinear_u(4,1,512));
    geo_net.addLayer('geo_deconv2', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'x30', 'x31', 'geo_deconv2_f');

    % Conv7
    geo_net.addLayer('geo_concat2', dagnn.Concat('dim',3), {'x31', 'x14'}, {'x32'});

    geo_conv7 = dagnn.Conv('size', [3 3 512+256 256], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv7', geo_conv7, {'x32'}, {'x33'}, {'geo_conv7_f','geo_conv7_b'});
    geo_net.addLayer('geo_relu7', dagnn.ReLU(), {'x33'}, {'x34'});

    geo_conv7b = dagnn.Conv('size', [3 3 256 256], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv7b', geo_conv7b, {'x34'}, {'x35'}, {'geo_conv7b_f','geo_conv7b_b'});
    geo_net.addLayer('geo_relu7b', dagnn.ReLU(), {'x35'}, {'x36'});    
    
    filters = single(bilinear_u(4,1,256));
    geo_net.addLayer('geo_deconv3', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'x36', 'x37', 'geo_deconv3_f');
    
    % Conv8
    geo_net.addLayer('geo_concat3', dagnn.Concat('dim',3), {'x37', 'x9'}, {'x38'});

    geo_conv8 = dagnn.Conv('size', [3 3 256+128 128], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv8', geo_conv8, {'x38'}, {'x39'}, {'geo_conv8_f','geo_conv8_b'});
    geo_net.addLayer('geo_relu8', dagnn.ReLU(), {'x39'}, {'x40'});

    geo_conv8b = dagnn.Conv('size', [3 3 128 128], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv8b', geo_conv8b, {'x40'}, {'x41'}, {'geo_conv8b_f','geo_conv8b_b'});
    geo_net.addLayer('geo_relu8b', dagnn.ReLU(), {'x41'}, {'x42'});

    filters = single(bilinear_u(4,1,128));
    geo_net.addLayer('geo_deconv4', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'x42', 'x43', 'geo_deconv4_f');

    % Conv9
    geo_net.addLayer('geo_concat4', dagnn.Concat('dim',3), {'x43', 'x4'}, {'x44'});

    geo_conv9 = dagnn.Conv('size', [3 3 128+64 64], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv9', geo_conv9, {'x44'}, {'x45'}, {'geo_conv9_f','geo_conv9_b'});
    geo_net.addLayer('geo_relu9', dagnn.ReLU(), {'x45'}, {'x46'});

    geo_conv9b = dagnn.Conv('size', [3 3 64 64], 'hasBias', true, 'pad', 1, 'stride', 1);
    geo_net.addLayer('geo_conv9b', geo_conv9b, {'x46'}, {'x47'}, {'geo_conv9b_f','geo_conv9b_b'});
    geo_net.addLayer('geo_relu9b', dagnn.ReLU(), {'x47'}, {'x48'});

    % Conv10- Geometric transformation estimation
    geo_conv10 = dagnn.Conv('size', [9 9 64 6], 'hasBias', true, 'pad', 4, 'stride', 1);
    geo_net.addLayer('geo_conv10', geo_conv10, {'x48'}, {'aff_transform'}, {'geo_conv10_f','geo_conv10_b'});
    
    %% Iterative Inference - #init  
    net.addLayer('init_corr_volume', Constraint_Correlation_Layer('max_disp',maxdisp,'stride',1), {'f1_x30_norm','f2_x30_norm'},{'init_corr_vol'});
    net.addLayer('init_norm', L2norm_Layer(), {'init_corr_vol'},{'init_corr_vol_norm'});        
    
    netStruct = net.saveobj;
    geo_netStruct = geo_net.saveobj;
    init_geo_netStruct = netNamePrefix(geo_netStruct,'init_','init_','');
    init_geo_netStruct = fuseNetStruct(netStruct,init_geo_netStruct);
    net = dagnn.DagNN.loadobj(init_geo_netStruct);
    
    net.addLayer('init_comp', dagnn.Sum(), {'init','init_aff_transform'}, {'comp_init_aff_transform'});
    
    %% Iterative Inference - #1
    net.addLayer('iter1_f2_transform', Transformer_Layer(), {'f2_x28','comp_init_aff_transform'}, {'iter1_f2_last1_input'}); 
    
    netStruct = net.saveobj;
    iter1_f2_last1_netStruct = netNamePrefix(net_vgg_last1.saveobj,'iter1_f2_last1_','iter1_f2_last1_','');
    iter1_f2_last1_netStruct = fuseNetStruct(netStruct,iter1_f2_last1_netStruct);
    net = dagnn.DagNN.loadobj(iter1_f2_last1_netStruct);    
    
    net.addLayer('iter1_corr_volume', Constraint_Correlation_Layer('max_disp',maxdisp,'stride',1), {'f1_x30_norm','iter1_f2_last1_norm'},{'iter1_corr_vol'});
    net.addLayer('iter1_norm', L2norm_Layer(), {'iter1_corr_vol'},{'iter1_corr_vol_norm'});        
    
    netStruct = net.saveobj;
    geo_netStruct = geo_net.saveobj;
    init_geo_netStruct = netNamePrefix(geo_netStruct,'iter1_','iter1_','');
    init_geo_netStruct = fuseNetStruct(netStruct,init_geo_netStruct);
    net = dagnn.DagNN.loadobj(init_geo_netStruct);

    net.addLayer('iter1_comp', dagnn.Sum(), {'comp_init_aff_transform','iter1_aff_transform'}, {'comp_iter1_aff_transform'});
	
    filters = single(bilinear_u(4,1,6));
	net.addLayer('iter1_deconv', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'comp_iter1_aff_transform', 'comp_iter1_aff_transform_', 'deconv_aff_x2'); 

    %% Iterative Inference - #2 
    net.addLayer('iter2_f2_transform', Transformer_Layer(), {'f2_x19','comp_iter1_aff_transform_'}, {'iter2_f2_last_input'}); 

    netStruct = net.saveobj;
    iter2_f2_last_netStruct = netNamePrefix(net_vgg_last.saveobj,'iter2_f2_last_','iter2_f2_last_','');
    iter2_f2_last_netStruct = fuseNetStruct(netStruct,iter2_f2_last_netStruct);
    net = dagnn.DagNN.loadobj(iter2_f2_last_netStruct);
    
    net.addLayer('iter2_corr_volume', Constraint_Correlation_Layer('max_disp',maxdisp,'stride',1), {'f1_x21_norm','iter2_f2_last_norm'},{'iter2_corr_vol'});
    net.addLayer('iter2_norm', L2norm_Layer(), {'iter2_corr_vol'},{'iter2_corr_vol_norm'});        
    
    netStruct = net.saveobj;
    geo_netStruct = geo_net.saveobj;
    iter2_geo_netStruct = netNamePrefix(geo_netStruct,'iter2_','iter2_','');
    iter2_geo_netStruct = fuseNetStruct(netStruct,iter2_geo_netStruct);
    net = dagnn.DagNN.loadobj(iter2_geo_netStruct);
    
    net.addLayer('iter2_comp', dagnn.Sum(), {'comp_iter1_aff_transform_','iter2_aff_transform'}, {'comp_iter2_aff_transform'});
    
    %% Iterative Inference - #3  
    net.addLayer('iter3_f2_transform', Transformer_Layer(), {'f2_x19','comp_iter2_aff_transform'}, {'iter3_f2_last_input'}); 

    netStruct = net.saveobj;
    iter3_f2_last_netStruct = netNamePrefix(net_vgg_last.saveobj,'iter3_f2_last_','iter3_f2_last_','');
    iter3_f2_last_netStruct = fuseNetStruct(netStruct,iter3_f2_last_netStruct);
    net = dagnn.DagNN.loadobj(iter3_f2_last_netStruct);
    
    net.addLayer('iter3_corr_volume', Constraint_Correlation_Layer('max_disp',maxdisp,'stride',1), {'f1_x21_norm','iter3_f2_last_norm'},{'iter3_corr_vol'});
    net.addLayer('iter3_norm', L2norm_Layer(), {'iter3_corr_vol'},{'iter3_corr_vol_norm'});        
    
    netStruct = net.saveobj;
    geo_netStruct = geo_net.saveobj;
    iter3_geo_netStruct = netNamePrefix(geo_netStruct,'iter3_','iter3_','');
    iter3_geo_netStruct = fuseNetStruct(netStruct,iter3_geo_netStruct);
    net = dagnn.DagNN.loadobj(iter3_geo_netStruct);
    
    net.addLayer('iter3_comp', dagnn.Sum(), {'comp_iter2_aff_transform','iter3_aff_transform'}, {'comp_iter3_aff_transform'});
    
    %% Iterative Inference - #4  
    net.addLayer('iter4_f2_transform', Transformer_Layer(), {'f2_x19','comp_iter3_aff_transform'}, {'iter4_f2_last_input'}); 

    netStruct = net.saveobj;
    iter4_f2_last_netStruct = netNamePrefix(net_vgg_last.saveobj,'iter4_f2_last_','iter4_f2_last_','');
    iter4_f2_last_netStruct = fuseNetStruct(netStruct,iter4_f2_last_netStruct);
    net = dagnn.DagNN.loadobj(iter4_f2_last_netStruct);
    
    net.addLayer('iter4_corr_volume', Constraint_Correlation_Layer('max_disp',maxdisp,'stride',1), {'f1_x21_norm','iter4_f2_last_norm'},{'iter4_corr_vol'});
    net.addLayer('iter4_norm', L2norm_Layer(), {'iter4_corr_vol'},{'iter4_corr_vol_norm'});        
    
    netStruct = net.saveobj;
    geo_netStruct = geo_net.saveobj;
    iter4_geo_netStruct = netNamePrefix(geo_netStruct,'iter4_','iter4_','');
    iter4_geo_netStruct = fuseNetStruct(netStruct,iter4_geo_netStruct);
    net = dagnn.DagNN.loadobj(iter4_geo_netStruct);
    
    net.addLayer('iter4_comp', dagnn.Sum(), {'comp_iter3_aff_transform','iter4_aff_transform'}, {'comp_iter4_aff_transform'});
    
    net.addLayer('Transformation_to_Flow', Transform_to_Flow_Layer(), {'comp_iter4_aff_transform'}, {'comp_iter4_flow_transform'});
    
	%% Upsampling
	filters = single(bilinear_u(4,1,2));
	net.addLayer('deconv1', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'comp_iter4_flow_transform', 'comp_iter4_flow_transform_x4', 'deconv_f_x2');      

	filters = single(bilinear_u(4,1,2));
	net.addLayer('deconv2', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'comp_iter4_flow_transform_x4', 'comp_iter4_flow_transform_x2', 'deconv_f_x2');      

	filters = single(bilinear_u(4,1,2));
	net.addLayer('deconv3', ...
        dagnn.ConvTranspose('size', size(filters), 'upsample', 2, 'crop', 1, 'hasBias', false), 'comp_iter4_flow_transform_x2', 'comp_iter4_flow_transform_x1', 'deconv_f_x2');      

    net.addLayer('image_transform', Flow_Layer(), {'img2','comp_iter4_flow_transform_x1'}, {'img2_transform'});
    
    %% Weakly-supervised Loss
    net.addLayer('objective1', Loss_wVisualization('loss', 'softmaxlog'), {'iter4_corr_vol','label','img1','img2_transform','img2'}, 'objective1');
    
    %%
    net.initParams();

    for i = 1:26
        net.params(i).value = net_vgg_first.params(i).value;
    end
    
    filters = single(bilinear_u(4,1,512));
    f = net.getParamIndex('geo_deconv1_f');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;

    filters = single(bilinear_u(4,1,512));
    f = net.getParamIndex('geo_deconv2_f');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;
    
    filters = single(bilinear_u(4,1,256));
    f = net.getParamIndex('geo_deconv3_f');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;

    filters = single(bilinear_u(4,1,128));
    f = net.getParamIndex('geo_deconv4_f');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;
     
    filters = single(bilinear_u(4,1,2));
    f = net.getParamIndex('deconv_f_x2');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;  
    
    filters = single(bilinear_u(4,1,6));
    f = net.getParamIndex('deconv_aff_x2');
    net.params(f).value = filters;
    net.params(f).learningRate = 0;
    net.params(f).weightDecay = 1;  

    net.params(net.getParamIndex('geo_conv10_f')).value = 0*net.params(net.getParamIndex('geo_conv10_f')).value;
    net.params(net.getParamIndex('geo_conv10_f')).learningRate = 0.1;
    net.params(net.getParamIndex('geo_conv10_b')).value = 0*net.params(net.getParamIndex('geo_conv10_b')).value;
    net.params(net.getParamIndex('geo_conv10_b')).learningRate = 0.1;
    
    if init_model 
        net_init = net;
        load('data/RTNs_synthetic/net-epoch.mat');
        net_pretrained = dagnn.DagNN.loadobj(net);
        for i = 1:70
            net_init.params(i).value = net_pretrained.params(i).value;
        end
        net = net_init;
    end
end
