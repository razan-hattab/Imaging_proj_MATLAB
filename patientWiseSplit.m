function [trainIdx, valIdx, testIdx] = patientWiseSplit(imgList, trainRatio, valRatio)
% PATIENTWISESPLIT Split dataset by patient to avoid data leakage
%
% Inputs:
%   imgList    - Cell array of image file paths
%   trainRatio - Fraction of patients for training (e.g., 0.7)
%   valRatio   - Fraction of patients for validation (e.g., 0.15)
%
% Outputs:
%   trainIdx - Indices of images for training set
%   valIdx   - Indices of images for validation set
%   testIdx  - Indices of images for test set
%
% Example:
%   [trainIdx, valIdx, testIdx] = patientWiseSplit(imgList, 0.7, 0.15);
%
% Note: This function groups all slices from the same patient together
%       to prevent data leakage between train/val/test sets.

    % Extract patient IDs from file paths
    % Example path: /path/to/TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif
    % Patient ID: TCGA_CS_4941_19960909
    
    N = length(imgList);
    patientIDs = cell(N, 1);
    
    for i = 1:N
        [~, name, ~] = fileparts(imgList{i});
        % Remove the slice number suffix (e.g., "_1", "_12")
        % Pattern: TCGA_XX_XXXX_XXXXXXXX_N.tif -> TCGA_XX_XXXX_XXXXXXXX
        parts = strsplit(name, '_');
        if length(parts) >= 4
            % Rejoin first 4 parts (TCGA_CS_4941_19960909)
            patientIDs{i} = strjoin(parts(1:4), '_');
        else
            % Fallback: use the whole name without the last part
            patientIDs{i} = strjoin(parts(1:end-1), '_');
        end
    end
    
    % Get unique patient IDs
    uniquePatients = unique(patientIDs);
    numPatients = length(uniquePatients);
    
    fprintf('Total images: %d\n', N);
    fprintf('Total patients: %d\n', numPatients);
    
    % Randomly shuffle patients
    rng(42);  % Set seed for reproducibility
    shuffledPatients = uniquePatients(randperm(numPatients));
    
    % Calculate split points
    numTrain = round(trainRatio * numPatients);
    numVal = round(valRatio * numPatients);
    numTest = numPatients - numTrain - numVal;
    
    fprintf('Patient split: Train=%d, Val=%d, Test=%d\n', numTrain, numVal, numTest);
    
    % Split patients
    trainPatients = shuffledPatients(1:numTrain);
    valPatients = shuffledPatients(numTrain+1:numTrain+numVal);
    testPatients = shuffledPatients(numTrain+numVal+1:end);
    
    % Map patient IDs back to image indices
    trainIdx = [];
    valIdx = [];
    testIdx = [];
    
    for i = 1:N
        if ismember(patientIDs{i}, trainPatients)
            trainIdx = [trainIdx; i];
        elseif ismember(patientIDs{i}, valPatients)
            valIdx = [valIdx; i];
        elseif ismember(patientIDs{i}, testPatients)
            testIdx = [testIdx; i];
        end
    end
    
    fprintf('Image split: Train=%d, Val=%d, Test=%d\n', ...
        length(trainIdx), length(valIdx), length(testIdx));
    
    % Verify no patient overlap
    trainPatientSet = unique(patientIDs(trainIdx));
    valPatientSet = unique(patientIDs(valIdx));
    testPatientSet = unique(patientIDs(testIdx));
    
    assert(isempty(intersect(trainPatientSet, valPatientSet)), ...
        'Data leakage: Train and Val share patients!');
    assert(isempty(intersect(trainPatientSet, testPatientSet)), ...
        'Data leakage: Train and Test share patients!');
    assert(isempty(intersect(valPatientSet, testPatientSet)), ...
        'Data leakage: Val and Test share patients!');
    
    fprintf('✓ No data leakage detected - all patients are in separate sets\n');
end
