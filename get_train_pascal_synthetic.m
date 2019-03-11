clc; clear; close all;
run('vlfeat-0.9.20\toolbox\vl_setup');
run('matconvnet-1.0-beta25\matlab\vl_setupnn.m');

%% PASCAL-VOC 2011 Database
BenchmarkPath = 'datasets\pascal-voc11';
if ~exist(BenchmarkPath)
    fprintf('Downloading PASCAL-VOC 2011 benchmark...\n') ;
    mkdir('datasets','pascal-voc11');
    urlwrite('http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar', ...
    fullfile('datasets','pascal-voc11','VOCtrainval_25-May-2011.tar')) ;
    fprintf('Unzipping...\n') ;
    untar(fullfile('datasets','pascal-voc11','VOCtrainval_25-May-2011.tar'),fullfile('datasets','pascal-voc11'));       
end

folder_name = 'datasets\pascal-voc11\TrainVal\VOCdevkit\VOC2011\JPEGImages';
folder_data = dir(folder_name);
folder_data = folder_data(3:end,:);

count_data = 0;

for ii = 1:length(folder_data)
	count_data = count_data + 1;
	image1 = imread(fullfile(folder_name,folder_data(ii).name));
    image1 = imresize(image1,[256,256]);
    
	featurePath = 'datasets\db_pascal_synthetic';
	save_folder_name = sprintf('%d',count_data);
	mkdir(fullfile(featurePath,save_folder_name));
    imwrite(uint8(image1),fullfile(fullfile(featurePath,save_folder_name),'image1.png'));
	data{count_data} = fullfile(featurePath,save_folder_name);   
end

imdb = struct;
imdb.images.data = data;
imdb.images.set = single(ones(1,length(data)));
save('data/imdb_pascal_synthetic','imdb','-v7.3');