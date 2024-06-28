
%% Get the Image Data 
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

%% Pre-process the data (ONLY RUN ONCE, AND IF THE DATA IS THE ORIGINAL, IT FREEZES MATLAB EVEN AFTER IT'S DONE)
% Get names in common
for name = imageNames
    % Get all files that are in the images dataset but not the label
    % dataset
    if  isempty(find(ismember(labelNames, name), 1))
        % Delete them.
        delete(fullfile("data_for_moodle/","images_256/",strcat(name,'.jpg')))
    else
        % Now combine labels that aren't background or flower
        label = imread(fullfile("data_for_moodle/","labels_256/",strcat(name,'.png')));
        label = removeLabels(label);
        imwrite(label,strcat("data_for_moodle/","labels_256/",strcat(name,'.png')))
    
        %Filter to remove compression artefacts
        image = imread(fullfile("data_for_moodle/","images_256/",strcat(name,'.jpg')));
        [noisyR,noisyG,noisyB] = imsplit(image);
        denoisedR = filter2(fspecial('disk',1),noisyR)/255;
        denoisedG = filter2(fspecial('disk',1),noisyG)/255;
        denoisedB = filter2(fspecial('disk',1),noisyB)/255;
        denoisedRGB = cat(3,denoisedR,denoisedG,denoisedB);
        imwrite(denoisedRGB,strcat("data_for_moodle/","images_256/",strcat(name,'.jpg')))
    end
    
end

%% Split and Setup Data

% Gets labels from ground truth data
classNames = ["flower", "background"];
labelIds = [1 3];

% Setup RNG for splitting data
rng(12345);


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


% Create combined Data (and also remove the labels that arent background
% and flower by setting them to background).
trainingData = combine(imdsTrain,pxdsTrain);
validationData = combine(imdsVal,pxdsVal);
testData = combine(imdsTest,pxdsTest);

% Create transformed data that has augmentations during training
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

%% Show an instance of the dataset

I = readimage(imdsTest,26);
L = readimage(pxdsTest,26);

t = imresize(I,5);
tl = imresize(uint16(L),5);

imshowpair(t,tl,'montage')

%% Setup the from-scratch network.

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
%load('checkpoints\net_checkpoint__3968__2024_05_04__17_18_44.mat','net')

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
    dropoutLayer(0.2)
    convolution2dLayer(1,2)
    softmaxLayer()
];

% Add the layers to the network
dnet = addLayers(dnet,layers);
deepNetworkDesigner(dnet)
% Setup options
opts = trainingOptions('sgdm',...
    'InitialLearnRate',1e-3,...
    'MaxEpochs',256,...
     'ValidationData',validationData, ...
    'ValidationFrequency',100, ...
    'ValidationPatience',15, ...
    'MiniBatchSize',16,  ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'OutputNetwork',"best-validation", ...
    "CheckpointPath","checkpoints");

%% Start training the network
[net2,info] = trainnet(transformedTrainingData,dnet,@modelLoss,opts);

%% Test Images in the Test Set for a specific saved network
netpre = coder.loadDeepLearningNetwork("trainTransform[-20 20],[-20 20],[1 1.5]2Class1DropoutBlur.mat");

testImage = readimage(imdsTest,54);
testLabel = readimage(pxdsTest,54);
figure
imshow(testImage);
C = semanticseg(testImage,netpre);
B = labeloverlay(testImage,C);
imshow(B)

%% Evaluate the specific saved network.
pxdsResults = semanticseg(imdsTest,netpre,Classes=classNames,WriteLocation=tempdir,MiniBatchSize=16);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest);
metrics.ClassMetrics
metrics.ConfusionMatrix

%% Setup Existing Network for Transfer Learning

[pretrainNet,classNamess] = imagePretrainedNetwork("resnet18");

% Resize image dataset
resizedData = transform(transformedTrainingData, @(x)resizeImage(x,[224 224]));
resizedVal = transform(validationData, @(x)resizeImage(x,[224 224]));
resizedImdsTest = imdsTest;
resizedImdsTest.ReadFcn = @(loc)imresize(imread(loc),[224 224]);
resizedPxdsTest = pxdsTest;
resizedPxdsTest.ReadFcn = @(loc)imresize(imread(loc),[224 224]);

% Freeze the first 10 layers
layers = pretrainNet.Layers;
layers(1:10) = slowWeights(pretrainNet.Layers(1:10));

% Recreate the network with the frozen layers.
pretrainNet = dlnetwork(createLgraphUsingConnections(layers,pretrainNet.Connections));

% Remove the classifier
pretrainNet = removeLayers(pretrainNet,{'fc1000' 'prob', 'pool5','res4a_branch2a','res4a_branch1','bn4a_branch2a','bn4a_branch1','res4a_branch2a_relu','res4a_branch2b','bn4a_branch2b','res4a','res4a_relu','res4b_branch2a','bn4b_branch2a','res4b_branch2a_relu','res4b_branch2b','bn4b_branch2b','res4b','res4b_relu','res5a_branch2a_relu','res5a_branch2b','bn5a_branch2b','res5a_branch1','bn5a_branch1','res5a','res5a_relu','res5b_branch2a','bn5b_branch2a','res5b_branch2a_relu','res5b_branch2b','bn5b_branch2b','res5a_branch2a','bn5a_branch2a','res5b','res5b_relu','res3a_branch2a','bn3a_branch2a','res3a_branch2a_relu','res3a_branch2b','bn3a_branch2b','res3a_branch1','bn3a_branch1','res3a','res3a_relu','res3b_branch2a','bn3b_branch2a','res3b_branch2a_relu','res3b_branch2b','bn3b_branch2b','res3b','res3b_relu'});

%pretrainNet = connectLayers(pretrainNet, 'res3b_relu', 'res5b_relu');

% Upsample network to use
upSample = [
        % upsample
    transposedConv2dLayer(2,512,'Stride',2,'Cropping',0,'Name','upsample')
    reluLayer()
    transposedConv2dLayer(2,256,'Stride',2,'Cropping',0)
    reluLayer()
    dropoutLayer(0.2)
    convolution2dLayer(1,2)
    softmaxLayer()];

% Add the upsampling layers
pretrainNet = addLayers(pretrainNet,upSample);
% Connect the upsampling layers together
pretrainNet = connectLayers(pretrainNet, 'res2b_relu', 'upsample');

%load('checkpoints\net_checkpoint__4032__2024_05_04__18_51_32.mat','net')

% Setup options
preopts = trainingOptions('sgdm',...
    'InitialLearnRate',3e-3,...
    'MaxEpochs',256,...
     'ValidationData',resizedVal, ...
    'ValidationFrequency',100, ...
    'MiniBatchSize',4,  ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'OutputNetwork',"best-validation", ...
    "CheckpointPath","checkpoints");
deepNetworkDesigner(pretrainNet)

%% Train network

[net3,info] = trainnet(resizedData,pretrainNet,@modelLoss,preopts);

%%

%% Test Images in the Test Set for a specific saved network
netpre = coder.loadDeepLearningNetwork("resnet2DropoutBlur.mat");

testImage = readimage(resizedImdsTest,54);
figure
imshow(testImage);
C = semanticseg(testImage,netpre);
B = labeloverlay(testImage,C);
imshow(B)

%% Evaluate the specific saved network.
pxdsResults = semanticseg(resizedImdsTest,netpre,Classes=classNames,WriteLocation=tempdir,MiniBatchSize=8);
metrics = evaluateSemanticSegmentation(pxdsResults,resizedPxdsTest);
metrics.ClassMetrics
metrics.ConfusionMatrix
deepNetworkDesigner(netpre)


%% Functions

% Removes the other labels from the label image so that only background and
% flower are present.
function x = removeLabels(data)

    % Apply resizing
   labelChannel = data(:, :, 1);

   % Get labels
   label2 = labelChannel == 2;
   label0 = labelChannel == 0;
   label4 = labelChannel == 4;

   % Now set these labels to background class 3
   labelChannel(label2) = 3;
   labelChannel(label4) = 3;
   % Boundaries are normally flowers to don't remove to background
   labelChannel(label0) = 1;

   % Recombine the image and return
   x = cat(1, labelChannel);
end

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


% Resizes an image
function data = resizeImage(data, size)

    % Apply resizing
    data{1} = imresize(data{1},size);
    data{2} = imresize(data{2},size);
end


% Semantic Segmentation Loss Function
function loss = modelLoss(Y,T) 
  mask = ~isnan(T);
  targets(isnan(T)) = 0;
  loss = crossentropy(Y,T,Mask=mask,NormalizationFactor="mask-included"); 
end

% Freezes the weights of the inputted layers.
% Should be in Matlab 2018a but has been removed for some reason, so is
% inserted manually.
function layers = slowWeights(layers)
    for i = 1:size(layers,1)
        props = properties(layers(i));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(i).(propName) = 0.4;
            end
        end
    end
end

% By https://github.com/LightingResearchCenter/GPMNet/blob/master/createLgraphUsingConnections.m
% Create a new layer with addlayers and connectlayers. 
% Should be in matlab but is not for some reason.
function lgraph = createLgraphUsingConnections(layers,connections)
    lgraph = layerGraph();
    for i = 1:numel(layers)
        lgraph = addLayers(lgraph,layers(i));
    end
    
    for c = 1:size(connections,1)
        lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
    end
end

%%
% Visualise network layers
layer =16;
name = netpre.Layers(layer).Name;

channels = 1:9;
I = deepDreamImage(netpre,name,channels,'PyramidLevels',1);


figure
I = imtile(I,'ThumbnailSize',[128 128]);
imshow(I)
