
%% Week 1-2: Data Curation and Preparation
clear; clc; close all;

%% 1. Acquire and Review the Dataset

% Load all image files
DATA_ROOT = fullfile(pwd, 'kaggle_3m');

allTifs = dir(fullfile(DATA_ROOT, '**', '*.tif'));
imagePaths = {}; maskPaths = {};
for i = 1:numel(allTifs)
    fname = allTifs(i).name;
    if contains(fname, '_mask', 'IgnoreCase', true), continue; end
    imgPath = fullfile(allTifs(i).folder, fname);
    [fd, base, ext] = fileparts(imgPath);
    mskPath = fullfile(fd, [base '_mask' ext]);
    if isfile(mskPath)
        imagePaths{end+1} = imgPath; %#ok<SAGROW>
        maskPaths{end+1}  = mskPath; %#ok<SAGROW>
    end
end
fprintf('Total image-mask pairs: %d\n', numel(imagePaths));

% Check first image 
img = imread(imagePaths{1}); % imshow(img);
fprintf('Image dimensions: %d x %d x %d\n', size(img, 1), size(img, 2), size(img, 3));

%% 2. Split Dataset

% Split into train/validation/test (70%/15%/15%)
N = numel(imagePaths);
rng(42); % For reproducibility
idx = randperm(N);

trainEnd = floor(0.7 * N);
valEnd = floor(0.85 * N);

trainIdx = idx(1:trainEnd);
valIdx = idx(trainEnd+1:valEnd);
testIdx = idx(valEnd+1:end);

trainImages = imagePaths(trainIdx); trainMasks = maskPaths(trainIdx);
valImages = imagePaths(valIdx); valMasks = maskPaths(valIdx);
testImages = imagePaths(testIdx); testMasks = maskPaths(testIdx);

fprintf('Train: %d, Validation: %d, Test: %d\n', length(trainImages), length(valImages), length(testImages));

%Save split 
save('dataset_split.mat', 'trainImages', 'trainMasks','valImages', 'valMasks', 'testImages', 'testMasks');

%% Clean & Preprocess Data (normalization, resizing, masking)
% - FLAIR ONLY (channel 2)
% - Resize to 256x256
% - Normalize images to [0,1]
% - Force masks to binary (0/1) with nearest-neighbor resize

IMG_SIZE = 256;

USE_3CH = true;
[X_train, y_train] = loadSet(trainImages, trainMasks, IMG_SIZE, USE_3CH);
[X_val,   y_val  ] = loadSet(valImages,   valMasks,   IMG_SIZE, USE_3CH);
[X_test,  y_test ] = loadSet(testImages,  testMasks,  IMG_SIZE, USE_3CH);

fprintf(['Preprocessed:\n' ...
         '  Train X:%s  y:%s\n' ...
         '  Val   X:%s  y:%s\n' ...
         '  Test  X:%s  y:%s\n'], ...
    mat2str(size(X_train)), mat2str(size(y_train)), ...
    mat2str(size(X_val)),   mat2str(size(y_val)),   ...
    mat2str(size(X_test)),  mat2str(size(y_test)));

%% 3) Data balancing and augmentation (train only) (rotation, flipping, scaling, noise)
% Balance by oversampling the minority (tumor-present vs non-tumor):
% - random flip
% - ±10° rotation
% - random isotropic scaling (0.9–1.1) with center crop/pad
% - light Gaussian noise (image only)

[X_train, y_train] = balanceWithAug(X_train, y_train, IMG_SIZE);

fprintf(['After balancing & augmentation:\n' ...
        '  Train X:%s  y:%s\n'] ...
        ,mat2str(size(X_train)), mat2str(size(y_train)));

%% 🔍 Filter & rebalance training slices (keep all tumor + 10% negatives)

% 1) Identify blank slices (mean intensity ~ 0) and tumor presence
nonBlank = arrayfun(@(i) mean(X_train(:,:,:,i),'all') > 0.01, 1:size(X_train,4));  % tweak 0.01 if needed
hasTumor = arrayfun(@(i) any(y_train(:,:,1,i),'all'), 1:size(y_train,4));

tumorIdx = find(nonBlank & hasTumor);            % valid, has tumor
negIdx   = find(nonBlank & ~hasTumor);           % valid, no tumor

% 2) Sample ~10% of negatives (hard negatives) with a floor to avoid zero
rng(42);                                         % reproducible sampling
negKeepCount = max( min(round(0.10 * numel(negIdx)), 5000),  min(300, numel(negIdx)) );
% ^ keeps 10% negatives; at least 300 if available; caps at 5000 to avoid explosion

negKeep   = randsample(negIdx, negKeepCount);

% 3) Final keep set = all tumor + sampled negatives
keepIdx   = sort([tumorIdx, negKeep]);           %#ok<AGROW>
X_train   = X_train(:,:,:,keepIdx);
y_train   = y_train(:,:,:,keepIdx);

% 4) Report
fprintf('After filtering:\n');
fprintf('  Tumor slices kept:   %d\n', numel(tumorIdx));
fprintf('  Negatives sampled:   %d / %d (%.1f%%)\n', numel(negKeep), numel(negIdx), 100*numel(negKeep)/max(1,numel(negIdx)));
fprintf('  Final training set:  X:%s  y:%s\n', mat2str(size(X_train)), mat2str(size(y_train)));

%  Save prepared tensors for Week-3 training
save('week1_2_prepared.mat','X_train','y_train','X_val','y_val','X_test','y_test','IMG_SIZE','-v7.3');

%% Helper Functions

function [X, Y] = loadSet(imgList, maskList, IMG_SIZE, use3ch)
% Load images/masks. If use3ch==true, keep all 3 MRI channels.
% If use3ch==false, use FLAIR-only (channel 2).
    if nargin < 4, use3ch = false; end

    N = numel(imgList);
    C = tern(use3ch, 3, 1);           % channels
    X = zeros(IMG_SIZE, IMG_SIZE, C, N, 'single');
    Y = false(IMG_SIZE, IMG_SIZE, 1, N);

    for i = 1:N
        I = imread(imgList{i});                 % 3-channel tif
        if use3ch
            I = imresize(I, [IMG_SIZE IMG_SIZE], 'bilinear');      % keep 3
            X(:,:,:,i) = im2single(I);
        else
            if size(I,3) >= 2, I = I(:,:,2); else, I = I(:,:,1); end  % FLAIR
            I = imresize(I, [IMG_SIZE IMG_SIZE], 'bilinear');
            X(:,:,1,i) = im2single(I);
        end

        M = imread(maskList{i});
        if size(M,3)>1, M = rgb2gray(M); end
        M = imresize(M>0, [IMG_SIZE IMG_SIZE], 'nearest');
        Y(:,:,1,i) = M;
    end
end

function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end

function [Iout, Mout] = augOnce(I, M, S)
% One paired augmentation: flip + rotate + scale + noise (image only)
    Iout = I; Mout = M;

    % Flip (50%)
    if rand < 0.5
        Iout = fliplr(Iout);
        Mout = fliplr(Mout);
    end

    % Rotate (±10°)
    ang  = -10 + 20*rand;
    Iout = imrotate(Iout, ang, 'bilinear', 'crop');   % preserves size
    Mout = imrotate(Mout, ang, 'nearest', 'crop') > 0;

    % Scale (0.9–1.1) with robust center crop/pad back to S×S
    s = 0.9 + 0.2*rand;

    if abs(s-1) > 1e-6
        Irs = imresize(Iout, s, 'bilinear');
        Mrs = imresize(Mout, s, 'nearest') > 0;

        [h,w,~] = size(Irs);

        if h >= S && w >= S
            % ---- center crop, but CLAMP indices to valid range ----
            y0 = floor((h - S)/2) + 1;
            x0 = floor((w - S)/2) + 1;
            y0 = max(1, min(y0, h - S + 1));
            x0 = max(1, min(x0, w - S + 1));

            Iout = Irs(y0:y0+S-1, x0:x0+S-1, :);
            Mout = Mrs(y0:y0+S-1, x0:x0+S-1);
        else
            % ---- pad into the center of an S×S canvas ----
            Ipad = zeros(S,S,size(Irs,3), 'like', Irs);
            Mpad = false(S,S,1);

            y0 = floor((S - h)/2) + 1;
            x0 = floor((S - w)/2) + 1;

            y1 = max(1, y0);  x1 = max(1, x0);
            y2 = min(S, y0 + h - 1);
            x2 = min(S, x0 + w - 1);

            srcY1 = 1 + (y1 - y0);  srcX1 = 1 + (x1 - x0);
            srcY2 = srcY1 + (y2 - y1);
            srcX2 = srcX1 + (x2 - x1);

            Ipad(y1:y2, x1:x2, :) = Irs(srcY1:srcY2, srcX1:srcX2, :);
            Mpad(y1:y2, x1:x2)    = Mrs(srcY1:srcY2, srcX1:srcX2);

            Iout = Ipad;  Mout = Mpad;
        end
    end

    % Light Gaussian noise on image only (keep mask binary)
    sigma = 0.02;
    Iout = Iout + sigma * randn(size(Iout), 'like', Iout);
    Iout = min(max(Iout, 0), 1);
end



% Week 3-4: Model Development
% we will use a U-Net for now as baseline. We can try smth more advanced
% like DeepLabV3+ later for more accuracy
load('week1_2_prepared.mat')

% Basic sanity
assert(ndims(X_train)==4 && ndims(y_train)==4, 'Expect 4-D arrays');
IMG_SIZE = size(X_train,1);
C        = size(X_train,3);   %
numTrain = size(X_train,4);
numVal   = size(X_val,4);
numTest  = size(X_test,4);

fprintf('training with image size %dx%dx%d | Train:%d Val:%d Test:%d\n', ...
    IMG_SIZE, IMG_SIZE, C, numTrain, numVal, numTest);

%% GPU check
execEnv = "cpu";
try
    g = gpuDevice; disp(g);
    execEnv = "gpu";
catch
    warning('GPU not available *throws tomatoes*');
end
fprintf('execution environment: %s\n', execEnv);

%% Build pixel labels and datastores (SEGMENTATION-FRIENDLY)

% Class names (keep same order everywhere)
classes = ["background","tumor"];
Y_train_cat = categorical(y_train, [false true], classes);
Y_val_cat   = categorical(y_val,   [false true], classes);

Y_train_cat_2D = squeeze(Y_train_cat);   % HxWxN
Y_val_cat_2D   = squeeze(Y_val_cat);

imdsTrain = arrayDatastore(X_train,       'IterationDimension', 4);
pxdsTrain = arrayDatastore(Y_train_cat_2D,'IterationDimension', 3);
dsTrain   = combine(imdsTrain, pxdsTrain);

imdsVal = arrayDatastore(X_val,       'IterationDimension', 4);
pxdsVal = arrayDatastore(Y_val_cat_2D,'IterationDimension', 3);
dsVal   = combine(imdsVal, pxdsVal);


% Class weights (from your masks) to help imbalance
tumorFrac = gather(nnz(y_train)) / numel(y_train);
w = 1 ./ [max(1e-6,1-tumorFrac), max(1e-6,tumorFrac)];
w = w / sum(w);

% Define U-Net
IMG_SIZE = size(X_train,1);
C        = size(X_train,3);          % now 3
imageSize  = [IMG_SIZE IMG_SIZE C];
numClasses = 2;

% Base network (use unetLayers because it returns a LayerGraph for trainNetwork)
lgraph = unetLayers(imageSize, numClasses, 'EncoderDepth', 4);

% Prefer Dice loss if available, else weighted CE
pxName = '';
for L = 1:numel(lgraph.Layers)
    if isa(lgraph.Layers(L),'nnet.cnn.layer.PixelClassificationLayer')
        pxName = lgraph.Layers(L).Name; break;
    end
end

if exist('dicePixelClassificationLayer','file') == 2
    % Dice loss helps with imbalance
    newPx = dicePixelClassificationLayer('Name','labels', ...
        'Classes',categorical(classes));
else
    % Weighted cross-entropy as fallback (heavy tumor weight)
    newPx = pixelClassificationLayer('Name','labels', ...
        'Classes',classes, 'ClassWeights',[0.2 0.8]);  % try [0.1 0.9] too
end
lgraph = replaceLayer(lgraph, pxName, newPx);


options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 30, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','auto', ...
    'ValidationData', dsVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots','training-progress');



%% train! woop woop
net = trainNetwork(dsTrain, lgraph, options);   


%% evaluate on the test set (Dice & IoU)
fprintf('\nEvaluating on test set...\n');
diceScores = zeros(1, numTest);
iouScores  = zeros(1, numTest);

for i = 1:numTest
    I = X_test(:,:,:,i);
    % predict categorical mask with class names matching "classes"
    Yp = semanticseg(I, net, 'ExecutionEnvironment','auto');

    % convert to logical for metrics: tumor=1, background=0
    Mgt = y_test(:,:,1,i);   % logical ground truth
    Mp  = (Yp == 'tumor');

    % dice = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN)
    tp  = nnz(Mp & Mgt);
    fp  = nnz(Mp & ~Mgt);
    fn  = nnz(~Mp & Mgt);
    diceScores(i) = 2*tp / max(1, 2*tp + fp + fn);
    iouScores(i)  = tp   / max(1, tp + fp + fn);
end

meanDice = mean(diceScores);
meanIoU  = mean(iouScores);
fprintf('mean dice: %.4f //// mean IoU: %.4f\n', meanDice, meanIoU);

%% 7) Visualize a few predictions
figure; 
for k = 1:4
    idx = randi(numTest);
    I = X_test(:,:,:,idx);
    Yp = semanticseg(I, net, 'ExecutionEnvironment','auto');
    Mp = (Yp == 'tumor');

    subplot(2,4,k);
    imshow(I(:,:,min(C,2)),[]); title(sprintf('Test %d: Input (ch %d)', idx, min(C,2)));

    subplot(2,4,k+4);
    imshowpair(I(:,:,min(C,2)), Mp); title('Prediction overlay');
end
fprintf('\nEvaluating on test set...\n');
numTest = size(X_test,4);
dice_all = zeros(1,numTest); iou_all  = zeros(1,numTest);
dice_pos = [];               iou_pos  = [];

for i = 1:numTest
    I  = X_test(:,:,:,i);
    Yp = semanticseg(I, net, 'ExecutionEnvironment','auto');
    Mp = (Yp=='tumor');
    Mp = bwareaopen(Mp, 30);           % remove tiny specks
    Mp = imfill(Mp, 'holes');          % fill small holes

    Mgt = y_test(:,:,1,i);             % logical GT

    tp = nnz(Mp & Mgt); fp = nnz(Mp & ~Mgt); fn = nnz(~Mp & Mgt);
    dice_all(i) = 2*tp / max(1,2*tp+fp+fn);
    iou_all(i)  = tp   / max(1,tp+fp+fn);

    if any(Mgt(:))
        dice_pos(end+1) = dice_all(i);
        iou_pos(end+1)  = iou_all(i);
    end
end

fprintf('Overall Dice (all slices): %.4f | IoU: %.4f\n', mean(dice_all), mean(iou_all));
fprintf('Dice (tumor slices only):  %.4f | IoU: %.4f | n=%d\n', ...
        mean(dice_pos), mean(iou_pos), numel(dice_pos));

% Pixel accuracy (optional)
acc_sum=0; pix_sum=0;
for i=1:numTest
    I  = X_test(:,:,:,i);
    Yp = semanticseg(I, net, 'ExecutionEnvironment','auto');
    pr = (Yp=='tumor'); gt = y_test(:,:,1,i);
    acc_sum = acc_sum + nnz(pr==gt); pix_sum = pix_sum + numel(gt);
end
fprintf('Pixel accuracy on TEST: %.2f%% | Tumor pixel fraction: %.4f\n', ...
        100*acc_sum/pix_sum, nnz(y_test)/numel(y_test));

save('week3_4_trained_unet.mat', ...
     'net','options','lgraph','imageSize','classes','USE_3CH', ...
     'dice_all','iou_all','dice_pos','iou_pos','-v7.3');
 function [XB, YB] = balanceWithAug(X, Y, S)
% Oversample minority class with paired augmentation (flip/rotate/scale/noise).
% Works for X with C = 1 or 3 channels; Y is single-channel logical.

    C = size(X,3);  % channel count

    % Detect tumor presence per slice
    hasTumor = squeeze(any(any(Y,1),2));
    hasTumor = reshape(hasTumor,1,[]);
    posIdx = find(hasTumor);
    negIdx = find(~hasTumor);
    nPos = numel(posIdx); 
    nNeg = numel(negIdx);

    % Already balanced?
    if nPos == nNeg
        p = randperm(size(X,4));
        XB = X(:,:,:,p); 
        YB = Y(:,:,:,p); 
        return;
    end

    % Decide which class to oversample
    target   = max(nPos, nNeg);
    minority = posIdx;
    if nPos > nNeg, minority = negIdx; end
    need     = target - min(nPos, nNeg);

    % Allocate augmented tensors with correct channels
    X_extra = zeros(S, S, C, need, 'like', X);
    Y_extra = false(S, S, 1, need);

    % Generate augmented samples
    for k = 1:need
        i0 = minority(randi(numel(minority)));
        [Ia, Ma] = augOnce(X(:,:,:,i0), Y(:,:,:,i0), S);
        X_extra(:,:,:,k) = Ia;
        Y_extra(:,:,:,k) = Ma;
    end

    % Concatenate and shuffle
    XB = cat(4, X, X_extra);
    YB = cat(4, Y, Y_extra);
    p = randperm(size(XB,4));
    XB = XB(:,:,:,p);
    YB = YB(:,:,:,p);
end
