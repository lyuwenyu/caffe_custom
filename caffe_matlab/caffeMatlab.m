
close all;
addpath('caffe/matlab/');

caffe.set_mode_gpu();
caffe.set_device(3);


model_file = 'model/VGG_ILSVRC_16_layers.caffemodel';
deploy_file = 'model/feature_net.prototxt';

% solver_file = '';
% xcnn = caffe.Solver(solver_file);
% xcnn.net.copy_from(model_file);
% xcnn.net.set_phase('train');

net = caffe.Net(deploy_file, model_file, 'test');


net.layer_names


mean_pix = [103.939, 116.779, 123.68];  %bgr
%imgp = '/home/xyy/Desktop/wenyu.lyu/workspace/mac1/datasets/123/4.jpg';
imgp = '1.jpg';
figure(), imshow(imread(imgp));
imwrite(imread(imgp), ['imgs/4.jpg']);
% img = imread(imgp);
% size(img)
% img = img(:, :, [3,2,1]); %rbg->bgr
% img = permute(img, [2,1,3]); %hwc -> whc
% size(img)
% img = single(img); %single
% img(:,:,1) = img(:,:,1) - mean_pix(1);
% img(:,:,2) = img(:,:,2) - mean_pix(2);
% img(:,:,3) = img(:,:,3) - mean_pix(3);

img = caffe.io.load_image(imgp);
for i=1:3
    img(:,:,i) = img(:,:,i) - mean_pix(i);
end
%imshow(img), hold on;

% 
% input_data(:,:,:,1) = img;
% input_data(:,:,:,2) = img;

net.set_input_dim([0,1,3,size(img,2),size(img,1)]);
size(net.blobs('data').get_data);


%net.forward({img});
net.forward({img});


xx = net.blobs('conv4_1').get_data();
size(xx);

for i = 1:20
    x = permute(xx(:,:,i,1),[2,1]);

    max1 = max(max(x,[],1),[],2);
    min1 = min(min(x,[],1),[],2);
    x = (x-min1) ./ (max1-min1);
    imwrite(x, ['imgs/conv4_1_' num2str(i,'%02d') '.jpg']);
end

caffe.reset_all();




