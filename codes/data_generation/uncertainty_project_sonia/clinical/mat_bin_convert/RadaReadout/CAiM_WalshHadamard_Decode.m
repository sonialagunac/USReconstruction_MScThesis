% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-static Walsh-Hadamard data decoding
% Sergio Sanabria, Dieter Schweizer
% 2018, CAiM Group, ETH Zürich
%
% The data input consist of Walsh-Hadamard coded multi-static receive data.
% It is organized as a single frame of data in an array of size:
% Receivebeam data x Nrx x Ntx
%
% The data are encoded according to the following schemes:
%  1:   bipolar matrix (+1, -1) 
%  2:   unipolar matrix Hn+ and Hn- 
%  3:   unipolar matrix Hn+ only
%
%  Input parameters:
%   order: order of Hadamard matrix H (=equal to number of elements in TX, e.g. 64
%   or 128)
%
%   codeType: 1 (bipolar), 2 (unipolar H+, H-), 3 (unipolar H+ only =
%   default)
%   See Lopez Villaverde, IEEE Trans. on ultrason., ferroel. and freq control, April 2017 
%
%  Output: Decoded data as an array of Decoded data x Nrx x Ntx
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Y] = CAiM_WalshHadamard_Decode(X, params)

% paramsH = struct('order', params.order, 'Nel', size(X,2), 'codeType', 1);
% Create bipolar Hadamard code matrix for decoding
% H = CAiM_WalshHadamard_Code(paramsH);
%% RR start
H = hadamard(128);
%% RR end

size_X = size(X);
% Reshape input data matrix to a (input data x Nrx) x Ntx matrix to have
% all RX data corresponding to one TX in one column
X = reshape(X,[size_X(1)*size_X(2) size(X,3)]);

switch (params.codeType)
% mode = 1: uses bipolar matrix (+1, -1)
% mode = 2: uses unipolar matrices Hn+, Hn- (horizontally concatenated)
% mode = 3: uses unipolar matrix Hn+

    case 1, % Bipolar Walsh Hadamard decoding
        Y = X*H/params.order;
        
    case 2, % Unipolar matrices Hn+, Hn- (horizontally concatenated)
        % We need to reorganize the data before the multiplication
        % X-dim3: [order+, order-, order+, order-, ..... ] noblocks:Nel/order
        nblocks = size_X(2)/params.order;
        consistencyMetric = zeros(size(X,1), nblocks);        
        indexp = zeros(size_X(3)/2, 1);
        indexn = indexp;
        for indexb = 1:nblocks, % For each sub-block (when Nel > orders)
            % Evaluate movement artifacts since first and last coded
            % TX was the same, then consistencyMetric gives the difference
            % between two "identical" TX-RX cycles
            consistencyMetric(:,indexb) = X(:,1 + (indexb - 1)*2*params.order) - X(:,2*params.order + (indexb - 1)*2*params.order);
            % Actual decoding: Y = 1/N * (Xplus -Xminus) * H
            X(:,2*params.order + (indexb - 1)*2*params.order) = 0;
            indexp((1:params.order) + (indexb - 1)*params.order) = (1:params.order) + (indexb - 1)*2*params.order;
            indexn((2:params.order) + (indexb - 1)*params.order) = (1:params.order - 1) + params.order + (indexb - 1)*2*params.order;
            indexn((1) + (indexb - 1)*params.order) = (params.order) + params.order + (indexb - 1)*2*params.order;
        end
        Y = (X(:,indexp)  - X(:,indexn))*H/params.order;     
                
    case 3, % Unipolar matrix Hn+
        % We need to reorganize the data before multiplication
        nblocks = size_X(2)/params.order;
        for indexb = 1:nblocks, % For each sub-block (when Nel > orders)
             % As implemented in LopezVillaverde2016-IEEE
             % Actual decoding: Y = 1/N * (2* Xplus - Xplus1) * H
             % Xplus1 is the received coded signal when applying the first
             % column of the Hplus matrix which is equal to a plane wave
             % since all TX elements are used
             X(:, 1 + (indexb - 1)*params.order: params.order + (indexb - 1)*params.order) = ...
             2*X(:, 1 + (indexb - 1)*params.order: params.order + (indexb - 1)*params.order) - X(:, 1 + (indexb - 1)*params.order)*ones(1,params.order);
             % [N1, N2] = [N1, N2] - [N1,1]*[1,N2] 

% S. Sanabria's solution: 28.11.2016
%               avgBlock = 2*sum(X(:, 1 + (indexb - 1)*params.order: params.order + (indexb - 1)*params.order), 2)/params.order;
%               backupX1 = X(:, 1 + (indexb - 1)*params.order);
%               X(:, 1 + (indexb - 1)*params.order: params.order + (indexb - 1)*params.order) = ...
%               2*X(:, 1 + (indexb - 1)*params.order: params.order + (indexb - 1)*params.order) - avgBlock*ones(1,params.order) ...
%               + X(:, params.order + 1 + (indexb - 1)*params.order)*ones(1,params.order);
              
        end        
        Y = X(:,1:end)*H/params.order; 
 %       Y = X(:,1:end-1)*H/params.order;       
end

% Reshape decoded data matrix back to a Decoded data x Nrx x Ntx matrix
Y = reshape(Y,[size_X(1) size_X(2) size(Y,2)]);

end

