
% Data Curation and Preparation
%% 1. Acquire and Review the Dataset

%1.1 Load and Organize Data
clear; clc; close all;

% Load all image files
DATA_ROOT = '/Users/razanhattab/Downloads/kaggle_3m'; 
allTifs = dir(fullfile(DATA_ROOT, '**', '*.tif'));
numFiles  = numel(allTifs);
fprintf('Total .tif files found: %d\n', numFiles);

% Match images with masks
imagePaths = {};
maskPaths  = {};

for i = 1:numFiles
    fname = allTifs(i).name;
    if contains(fname, '_mask', 'IgnoreCase', true)
        continue; % skip mask files; start from the image
    end

    imgPath = fullfile(allTifs(i).folder, fname);
    [folder, base, ext] = fileparts(imgPath);
    expectedMaskPath = fullfile(folder, [base '_mask' ext]);

    if isfile(expectedMaskPath)
        imagePaths{end+1} = imgPath;          
        maskPaths{end+1}  = expectedMaskPath;
    end
end

% Verification
fprintf('Images: %d, Masks: %d\n', length(imagePaths), length(maskPaths));
fprintf('Total image-mask pairs: %d\n', length(imagePaths));
% Show first 3 pairs
for i = 1:3
    [~, imgName] = fileparts(imagePaths{i});
    [~, maskName] = fileparts(maskPaths{i});
    fprintf('%s -> %s\n', imgName, maskName);
end

%1.2 Explore Dataset Structure

% Check image dimensions (from second image)
img = imread(imagePaths{1});
imshow(img)
fprintf('Image dimensions: %d x %d x %d\n', size(img, 1), size(img, 2), size(img, 3));
%{ 
Verify all 3 channels present
numChannels = size(img, 3);
if numChannels == 3
    fprintf('All images have 3 channels\n');
else
    fprintf('Warning: Images have %d channels\n', numChannels);
end
%}

%% 2. Clean & Preprocess Data

% Check size for both images and masks
needsResize = false;
for i = 1:length(imagePaths)
    img = imread(imagePaths{i});
    mask = imread(maskPaths{i});

    if ~isequal([size(img,1), size(img,2)], [256, 256]) || ~isequal([size(mask,1), size(mask,2)], [256, 256])
        needsResize = true;
        break;
    end
end

if needsResize
    fprintf('Some images need resizing\n');
else
    fprintf('All images are 256x256, no resizing needed\n');
end
% Normalize image to [0,1]
imgNorm = double(img) / 255;


%% 3. Split Dataset

% Split into train/validation/test (70%/15%/15%)
numSamples = length(imagePaths);
rng(42); % For reproducibility
idx = randperm(numSamples);

trainEnd = floor(0.7 * numSamples);
valEnd = floor(0.85 * numSamples);

trainIdx = idx(1:trainEnd);
valIdx = idx(trainEnd+1:valEnd);
testIdx = idx(valEnd+1:end);

trainImages = imagePaths(trainIdx);
trainMasks = maskPaths(trainIdx);

valImages = imagePaths(valIdx);
valMasks = maskPaths(valIdx);

testImages = imagePaths(testIdx);
testMasks = maskPaths(testIdx);

fprintf('Train: %d, Validation: %d, Test: %d\n', length(trainImages), length(valImages), length(testImages));

% Save split save('dataset_split.mat', 'trainImages', 'trainMasks','valImages', 'valMasks', 'testImages', 'testMasks');

%% 4. Data Balancing & Augmentation

