clc; clear; close all;
run('vlfeat-0.9.20\toolbox\vl_setup');
run('matconvnet-1.0-beta25\matlab\vl_setupnn.m');

%% PF-PASCAL Database
imdbPath = fullfile('data', 'PF-PASCAL.mat');
pf_pascal = load(imdbPath);
pf_pascal = pf_pascal.data;

count_data = 0;

for ii = 1:length(pf_pascal.images)
    count_data = count_data + 1; 
	image1 = pf_pascal.images{ii,1};
	image2 = pf_pascal.images{ii,2};
    image1_mask = zeros(224,224,'single');
    image2_mask = zeros(224,224,'single');
    bb1 = round(pf_pascal.bbox{ii,1});
    bb2 = round(pf_pascal.bbox{ii,2});
    image1_mask(bb1(2):bb1(4),bb1(1):bb1(3)) = 1;
    image2_mask(bb2(2):bb2(4),bb2(1):bb2(3)) = 1;
    
	image1 = imresize(image1,[256,256]);
	image2 = imresize(image2,[256,256]);
	image1_mask = imresize(single(image1_mask~=0),[256,256],'nearest');    
    
    mask1 = ones(32,32,'single')*41.*imresize(image1_mask,[32,32],'nearest');     

	featurePath = 'datasets\db_pf_pascal';
	save_folder_name = sprintf('%d',count_data);
	mkdir(fullfile(featurePath,save_folder_name));
    imwrite(uint8(image1),fullfile(fullfile(featurePath,save_folder_name),'image1.png'));
    imwrite(uint8(image2),fullfile(fullfile(featurePath,save_folder_name),'image2.png'));
    imwrite(uint8(mask1),fullfile(fullfile(featurePath,save_folder_name),'mask1.png'));
	data{count_data} = fullfile(featurePath,save_folder_name);   
    set(count_data) = pf_pascal.set(ii);
end

for ii = 1:length(pf_pascal.images)
    count_data = count_data + 1; 
	image1 = pf_pascal.images{ii,1};
	image2 = pf_pascal.images{ii,2};
    image1_mask = zeros(224,224,'single');
    image2_mask = zeros(224,224,'single');
    bb1 = round(pf_pascal.bbox{ii,1});
    bb2 = round(pf_pascal.bbox{ii,2});
    image1_mask(bb1(2):bb1(4),bb1(1):bb1(3)) = 1;
    image2_mask(bb2(2):bb2(4),bb2(1):bb2(3)) = 1;
    
	image1 = imresize(image1,[256,256]);
	image2 = imresize(image2,[256,256]);
	image1_mask = imresize(single(image1_mask~=0),[256,256],'nearest');    
	image1 = fliplr(image1);
	image2 = fliplr(image2);
	image1_mask = fliplr(image1_mask);
    
    mask1 = ones(32,32,'single')*41.*imresize(image1_mask,[32,32],'nearest');     

	featurePath = 'datasets\db_pf_pascal';
	save_folder_name = sprintf('%d',count_data);
	mkdir(fullfile(featurePath,save_folder_name));
    imwrite(uint8(image1),fullfile(fullfile(featurePath,save_folder_name),'image1.png'));
    imwrite(uint8(image2),fullfile(fullfile(featurePath,save_folder_name),'image2.png'));
    imwrite(uint8(mask1),fullfile(fullfile(featurePath,save_folder_name),'mask1.png'));
	data{count_data} = fullfile(featurePath,save_folder_name);  
    set(count_data) = pf_pascal.set(ii);
end

for ii = 1:length(pf_pascal.images)
    count_data = count_data + 1; 
	image1 = pf_pascal.images{ii,2};
	image2 = pf_pascal.images{ii,1};
    image1_mask = zeros(224,224,'single');
    image2_mask = zeros(224,224,'single');
    bb1 = round(pf_pascal.bbox{ii,2});
    bb2 = round(pf_pascal.bbox{ii,1});
    image1_mask(bb1(2):bb1(4),bb1(1):bb1(3)) = 1;
    image2_mask(bb2(2):bb2(4),bb2(1):bb2(3)) = 1;
    
	image1 = imresize(image1,[256,256]);
	image2 = imresize(image2,[256,256]);
	image1_mask = imresize(single(image1_mask~=0),[256,256],'nearest');    
    
    mask1 = ones(32,32,'single')*41.*imresize(image1_mask,[32,32],'nearest');     

	featurePath = 'datasets\db_pf_pascal';
	save_folder_name = sprintf('%d',count_data);
	mkdir(fullfile(featurePath,save_folder_name));
    imwrite(uint8(image1),fullfile(fullfile(featurePath,save_folder_name),'image1.png'));
    imwrite(uint8(image2),fullfile(fullfile(featurePath,save_folder_name),'image2.png'));
    imwrite(uint8(mask1),fullfile(fullfile(featurePath,save_folder_name),'mask1.png'));
	data{count_data} = fullfile(featurePath,save_folder_name); 
    set(count_data) = pf_pascal.set(ii);
end

for ii = 1:length(pf_pascal.images)
    count_data = count_data + 1; 
	image1 = pf_pascal.images{ii,2};
	image2 = pf_pascal.images{ii,1};
    image1_mask = zeros(224,224,'single');
    image2_mask = zeros(224,224,'single');
    bb1 = round(pf_pascal.bbox{ii,2});
    bb2 = round(pf_pascal.bbox{ii,1});
    image1_mask(bb1(2):bb1(4),bb1(1):bb1(3)) = 1;
    image2_mask(bb2(2):bb2(4),bb2(1):bb2(3)) = 1;
    
	image1 = imresize(image1,[256,256]);
	image2 = imresize(image2,[256,256]);
	image1_mask = imresize(single(image1_mask~=0),[256,256],'nearest');    
	image1 = fliplr(image1);
	image2 = fliplr(image2);
	image1_mask = fliplr(image1_mask);
    
    mask1 = ones(32,32,'single')*41.*imresize(image1_mask,[32,32],'nearest');     

	featurePath = 'datasets\db_pf_pascal';
	save_folder_name = sprintf('%d',count_data);
	mkdir(fullfile(featurePath,save_folder_name));
    imwrite(uint8(image1),fullfile(fullfile(featurePath,save_folder_name),'image1.png'));
    imwrite(uint8(image2),fullfile(fullfile(featurePath,save_folder_name),'image2.png'));
    imwrite(uint8(mask1),fullfile(fullfile(featurePath,save_folder_name),'mask1.png'));
	data{count_data} = fullfile(featurePath,save_folder_name);  
    set(count_data) = pf_pascal.set(ii);
end

imdb = struct;
imdb.images.data = data;
imdb.images.set = set;
save('data/imdb_pf_pascal.mat','imdb','-v7.3');