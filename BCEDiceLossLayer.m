classdef BCEDiceLossLayer < nnet.layer.RegressionLayer
    % BCEDICELOSS Combined Binary Cross-Entropy and Dice Loss Layer
    %
    % This layer combines BCE loss (for pixel-wise accuracy and sharp boundaries)
    % with Dice loss (for region overlap and class imbalance handling).
    %
    % Loss = BCE Loss + Dice Loss
    %
    % Usage:
    %   lossLayer = BCEDiceLossLayer('Name', 'bce_dice_loss');
    
    properties
        % Smoothing factor to prevent division by zero in Dice loss
        Smooth = 1.0
        
        % Weight for BCE loss (default: 1.0)
        BCEWeight = 1.0
        
        % Weight for Dice loss (default: 1.0)
        DiceWeight = 1.0
    end
    
    methods
        function layer = BCEDiceLossLayer(varargin)
            % Constructor
            % Parse name-value pairs
            p = inputParser;
            addParameter(p, 'Name', 'bce_dice_loss');
            addParameter(p, 'Smooth', 1.0);
            addParameter(p, 'BCEWeight', 1.0);
            addParameter(p, 'DiceWeight', 1.0);
            parse(p, varargin{:});
            
            % Set layer name
            layer.Name = p.Results.Name;
            layer.Smooth = p.Results.Smooth;
            layer.BCEWeight = p.Results.BCEWeight;
            layer.DiceWeight = p.Results.DiceWeight;
            
            % Set layer description
            layer.Description = sprintf('Combined BCE + Dice Loss (BCE weight: %.2f, Dice weight: %.2f)', ...
                layer.BCEWeight, layer.DiceWeight);
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Y - Predictions from network (H x W x C x N)
            % T - Ground truth targets (H x W x C x N)
            %
            % For binary segmentation:
            %   C = 2 (background, tumor)
            %   Y(:,:,1,:) = background probabilities
            %   Y(:,:,2,:) = tumor probabilities
            %   T(:,:,1,:) = background ground truth (0 or 1)
            %   T(:,:,2,:) = tumor ground truth (0 or 1)
            
            % Extract tumor channel (channel 2)
            Y_tumor = Y(:,:,2,:);  % Predicted tumor probabilities
            T_tumor = T(:,:,2,:);  % Ground truth tumor mask
            
            % Flatten to vectors for easier computation
            Y_flat = Y_tumor(:);
            T_flat = T_tumor(:);
            
            % Clamp predictions to avoid log(0)
            epsilon = 1e-7;
            Y_flat = max(min(Y_flat, 1 - epsilon), epsilon);
            
            %% 1. Binary Cross-Entropy Loss
            % BCE = -[y*log(p) + (1-y)*log(1-p)]
            bce_loss = -mean(T_flat .* log(Y_flat) + (1 - T_flat) .* log(1 - Y_flat));
            
            %% 2. Dice Loss
            % Dice = 1 - (2 * |Y ∩ T| + smooth) / (|Y| + |T| + smooth)
            intersection = sum(Y_flat .* T_flat);
            union = sum(Y_flat) + sum(T_flat);
            dice_coef = (2 * intersection + layer.Smooth) / (union + layer.Smooth);
            dice_loss = 1 - dice_coef;
            
            %% 3. Combined Loss
            loss = layer.BCEWeight * bce_loss + layer.DiceWeight * dice_loss;
        end
        
        function dLdY = backwardLoss(layer, Y, T)
            % Backward pass: compute gradient of loss w.r.t. predictions
            %
            % Y - Predictions from network (H x W x C x N)
            % T - Ground truth targets (H x W x C x N)
            %
            % Returns:
            % dLdY - Gradient of loss w.r.t. Y (same size as Y)
            
            % Get dimensions
            [H, W, C, N] = size(Y);
            
            % Initialize gradient
            dLdY = zeros(size(Y), 'like', Y);
            
            % Extract tumor channel
            Y_tumor = Y(:,:,2,:);
            T_tumor = T(:,:,2,:);
            
            % Flatten
            Y_flat = Y_tumor(:);
            T_flat = T_tumor(:);
            
            % Clamp predictions
            epsilon = 1e-7;
            Y_flat = max(min(Y_flat, 1 - epsilon), epsilon);
            
            %% Gradient of BCE Loss
            % d(BCE)/dY = -(y/p - (1-y)/(1-p)) / N
            dBCE_dY = -(T_flat ./ Y_flat - (1 - T_flat) ./ (1 - Y_flat)) / numel(Y_flat);
            
            %% Gradient of Dice Loss
            % d(Dice)/dY = -2 * (T * (|Y| + |T|) - Y * 2|T|) / (|Y| + |T|)^2
            intersection = sum(Y_flat .* T_flat);
            union = sum(Y_flat) + sum(T_flat);
            
            numerator = 2 * intersection + layer.Smooth;
            denominator = union + layer.Smooth;
            
            % Gradient for each element
            dDice_dY = -2 * (T_flat * denominator - Y_flat * 2 * sum(T_flat)) / (denominator^2);
            
            %% Combined Gradient
            dL_dY_flat = layer.BCEWeight * dBCE_dY + layer.DiceWeight * dDice_dY;
            
            % Reshape back to original dimensions and assign to tumor channel
            dLdY(:,:,2,:) = reshape(dL_dY_flat, [H, W, 1, N]);
            
            % Background channel gradient (channel 1) is typically set to zero
            % or computed symmetrically, but for simplicity we leave it as zero
            % since we're only optimizing the tumor channel
        end
    end
end
