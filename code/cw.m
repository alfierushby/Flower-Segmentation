
%% Pre-process the data

dataDir = fullfile("data_for_moodle");
imageDir = fullfile(dataDir,'images_256');
labelDir = fullfile(dataDir,'labels_256');

% First get the images to remove.
imagesSearch = dir(imageDir);
imagesSearch = imagesSearch(~ismember({imagesSearch.name},{'.','..'}));

labelsSearch = dir(labelDir);
labelsSearch = labelsSearch(~ismember({labelsSearch.name},{'.','..'}));

% Get label and image names:
labelNames = erase(string({labelsSearch.name}),'.png'); 
imageNames = erase(string({imagesSearch.name}),'.jpg');

% Get names in common and
for name = imageNames
    % Get all files that are in the images dataset but not the label
    % dataset
    if  isempty(find(ismember(labelNames, name), 1))
        % Delete them.
        delete(fullfile("data_for_moodle/","images_256/",strcat(name,'.jpg')))
    end
end

% Gets labels from ground truth data
classNames = ["flower", "background"];
labelIds = [1 3];

% Setup RNG for splitting data
rng(12345);

% Augment data
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3],...
    'RandScale',[0.5 1], ...
    'RandXReflection',true);


% Create initial datasets
imds = imageDatastore(imageDir);
pxds = pixelLabelDatastore(labelDir,classNames,labelIds);


% Get number of files and then create an array of random indexes
numFiles = numpartitions(imds);
indexes = randperm(numFiles);

% Set 60% of the data for training.
bound60 = round(numFiles*.6);
imdsTrain = subset(imds,indexes(1:bound60));
pxdsTrain = subset(pxds,indexes(1:bound60));

% Set 20% of the data for validation
bound20 = round(numFiles*.2);
imdsVal = subset(imds,indexes(bound60+1:bound60+bound20));
pxdsVal = subset(pxds,indexes(bound60+1:bound60+bound20));

% Set the rest for testing
imdsTest = subset(imds,indexes(bound60+bound20+1:numFiles));
pxdsTest = subset(pxds,indexes(bound60+bound20+1:numFiles));

%% Show an instance of the dataset

I = readimage(imdsTest,26);
L = readimage(pxdsTest,26);

t = imresize(I,5);
tl = imresize(uint16(L),5);

imshowpair(t,tl,'montage')

%% Train the from-scratch network.

% Create combined Data
trainingData = combine(imdsTrain,pxdsTrain);
validationData = combine(imdsVal,pxdsVal);
testData = combine(imdsTest,pxdsTest);

% Count labels
tbl = countEachLabel(pxds);

% Get Weights
totalNumberOfPixels = sum(tbl.PixelCount);
freq = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./freq;

% Make layers
filterSize = 3;
numFilters = 20;

% Setup checkpoint to train from
load('checkpoints\net_checkpoint__3472__2024_05_03__20_19_25.mat','net')

% Create a dlnetwork for populating with layers.
dnet = dlnetwork;

layers = [
    % downsample
    imageInputLayer([256 256 3])
    convolution2dLayer(filterSize,numFilters, 'Padding',1)
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,2*numFilters, 'Padding',1)
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,4*numFilters, 'Padding',1)
    batchNormalizationLayer()
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)

    % upsample
    transposedConv2dLayer(4,numFilters*4,'Stride',2,'Cropping',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters*2,'Stride',2,'Cropping',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1)
    reluLayer()
    convolution2dLayer(1,2)
    softmaxLayer()
];

% Add the layers to the network
dnet = addLayers(dnet,layers);

% Setup options
opts = trainingOptions('sgdm',...
    'InitialLearnRate',1e-3,...
    'MaxEpochs',128,...
     'ValidationData',validationData, ...
    'ValidationFrequency',100, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',16,  ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'OutputNetwork',"best-validation", ...
    "CheckpointPath","checkpoints");

%%
transformedTrainingData = transform(trainingData, @(x)augmentImageAndLabel(x,[-10 10],[-10 10],[1 1.5]));

% Display a portion of the transformed data
data = readall(transformedTrainingData);
rgb = cell(9,1);
for k = 1:9
    I = data{k,1};
    C = data{k,2};
    rgb{k} = labeloverlay(I,C);
end
montage(rgb)
%%
% Start training the network
[net2,info] = trainnet(transformedTrainingData,dnet,@modelLoss,opts);

%% Test Images in the Test Set for a specific saved network
netpre = coder.loadDeepLearningNetwork("trainTransform[-10 10],[-10 10],[1 1.5].mat");

testImage = readimage(imdsTest,30);
figure
imshow(testImage);
C = semanticseg(testImage,netpre);
B = labeloverlay(testImage,C);
imshow(B)

%% Evaluate the specific saved network.
pxdsResults = semanticseg(imdsTest,netpre,Classes=classNames,WriteLocation=tempdir);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
metrics.ClassMetrics
metrics.ConfusionMatrix
deepNetworkDesigner(netpre)

%% Train an existing network

[pretrainNet,classNames] = imagePretrainedNetwork("resnet18");
pretrainNet = removeLayers(pretrainNet,{'fc1000' 'prob'});

upSample = [
        % upsample
    transposedConv2dLayer(4,512,'Stride',2,'Cropping',1,'Name','upsample')
    reluLayer()
    transposedConv2dLayer(4,256,'Stride',2,'Cropping',1)
    reluLayer()
    convolution2dLayer(1,2)
    softmaxLayer()];

pretrainNet = freezeParameters(pretrainNet,[]);
pretrainNet = addLayers(pretrainNet,upSample);
pretrainNet = connectLayers(pretrainNet, 'pool5', 'upsample');
deepNetworkDesigner(pretrainNet)




%% Functions

% Augment images with scaling, translations, reflections, and colour
% modifications.
function data = augmentImageAndLabel(data, xTrans, yTrans, scale)


    % Apply colour jitter
    %data{1} = jitterColorHSV(data{1},"Brightness",0.2,"Contrast",0.3,"Saturation",0.1);
    
    tform = randomAffine2d(...
        XReflection=true,...
        XTranslation=xTrans, ...
        YTranslation=yTrans, ...
        Scale=scale);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{1}), tform, BoundsStyle='centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{1} = imwarp(data{1}, tform, OutputView=rout);
    data{2} = imwarp(data{2}, tform, OutputView=rout);
end


% Semantic Segmentation Loss Function
function loss = modelLoss(Y,T) 
  mask = ~isnan(T);
  targets(isnan(T)) = 0;
  loss = crossentropy(Y,T,Mask=mask,NormalizationFactor="mask-included"); 
end