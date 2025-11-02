
%% Week 1-2: Data Curation and Preparation
clear; clc; close all;

%% 1. Acquire and Review the Dataset

% Load all image files
DATA_ROOT = '/Users/razanhattab/Downloads/kaggle_3m'; 

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

% Save split 
%save('dataset_split.mat', 'trainImages', 'trainMasks','valImages', 'valMasks', 'testImages', 'testMasks');

%% Clean & Preprocess Data (normalization, resizing, masking)
% - FLAIR ONLY (channel 2)
% - Resize to 256x256
% - Normalize images to [0,1]
% - Force masks to binary (0/1) with nearest-neighbor resize

IMG_SIZE = 256;

[X_train, y_train] = loadSet(trainImages, trainMasks, IMG_SIZE);
[X_val,   y_val  ] = loadSet(valImages,   valMasks,   IMG_SIZE);
[X_test,  y_test ] = loadSet(testImages,  testMasks,  IMG_SIZE);

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

%(Optional) Save prepared tensors for Week-3 training
% save('week1_2_prepared.mat','X_train','y_train','X_val','y_val','X_test','y_test','IMG_SIZE','-v7.3');

%% Helper Functions

function [X, Y] = loadSet(imgList, maskList, IMG_SIZE)
% FLAIR extraction (channel 2), resize, normalize, binary masks
    N = numel(imgList);
    X = zeros(IMG_SIZE, IMG_SIZE, 1, N, 'single');
    Y = false(IMG_SIZE, IMG_SIZE, 1, N);
    for i = 1:N
        I = imread(imgList{i});                 % 3-channel tif (modalities)
        if size(I,3) >= 2, I = I(:,:,2); else, I = I(:,:,1); end  % FLAIR
        I = imresize(I, [IMG_SIZE IMG_SIZE], 'bilinear');
        X(:,:,1,i) = im2single(I);              % normalize [0,1]

        M = imread(maskList{i});                % 1-channel expected
        if size(M,3)>1, M = rgb2gray(M); end
        M = imresize(M>0, [IMG_SIZE IMG_SIZE], 'nearest');        % binary 0/1
        Y(:,:,1,i) = M;
    end
end

function [XB, YB] = balanceWithAug(X, Y, S)
% Oversample minority class with paired augmentation (flip/rotate/scale/noise)
    hasTumor = squeeze(any(any(Y,1),2)); hasTumor = reshape(hasTumor,1,[]);
    posIdx = find(hasTumor); negIdx = find(~hasTumor);
    nPos = numel(posIdx); nNeg = numel(negIdx);

    if nPos == nNeg
        p = randperm(size(X,4)); XB = X(:,:,:,p); YB = Y(:,:,:,p); return;
    end

    target   = max(nPos, nNeg);
    minority = posIdx; if nPos > nNeg, minority = negIdx; end
    need     = target - min(nPos, nNeg);

    X_extra = zeros(S,S,1,need,'like',X);
    Y_extra = false(S,S,1,need);
    for k = 1:need
        i0 = minority(randi(numel(minority)));
        [Ia, Ma] = augOnce(X(:,:,:,i0), Y(:,:,:,i0), S);
        X_extra(:,:,:,k) = Ia;  Y_extra(:,:,:,k) = Ma;
    end

    XB = cat(4, X, X_extra);
    YB = cat(4, Y, Y_extra);

    p = randperm(size(XB,4));
    XB = XB(:,:,:,p);
    YB = YB(:,:,:,p);
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
    Iout = imrotate(Iout, ang, 'bilinear', 'crop');
    Mout = imrotate(Mout, ang, 'nearest', 'crop') > 0;

    % Scale (0.9–1.1) with center crop/pad back to S×S
    s = 0.9 + 0.2*rand;
    if abs(s-1) > 1e-6
        Irs = imresize(Iout, s, 'bilinear');
        Mrs = imresize(Mout, s, 'nearest') > 0;
        [h,w] = size(Irs);
        if h >= S && w >= S
            y0 = floor((h - S)/2) + 1; x0 = floor((w - S)/2) + 1;
            Iout = Irs(y0:y0+S-1, x0:x0+S-1);
            Mout = Mrs(y0:y0+S-1, x0:x0+S-1);
        else
            Ipad = zeros(S,S,1,'like',Iout); Mpad = false(S,S,1);
            y0 = floor((S - h)/2) + 1; x0 = floor((S - w)/2) + 1;
            Ipad(y0:y0+h-1, x0:x0+w-1) = Irs;
            Mpad(y0:y0+h-1, x0:x0+w-1) = Mrs;
            Iout = Ipad; Mout = Mpad;
        end
    end

    % Light Gaussian noise on image only (keep mask binary)
    sigma = 0.02;
    Iout = Iout + sigma * randn(size(Iout), 'like', Iout);
    Iout = min(max(Iout, 0), 1);
end

%% Week 3-4: Model Development
