% -----------------------------------------------------------------------
%   load small patch of (ws x ws) from CIFAR-10 dataset
% -----------------------------------------------------------------------

function [xtr, ytr, patch, M, P] = load_patches(npatch, ws)

prepare_cifar10;

% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

xtr = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
ytr = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!
clear f1 f2 f3 f4 f5;
fname = sprintf('cifar10_ws_%d',ws);

if ~exist('patch','dir'),
    mkdir('patch');
end

try
    load(sprintf('patch/%s.mat', fname),'patch', 'M', 'P');
catch
    if npatch > 0,
        % extract random patch
        patch = zeros(npatch, ws*ws*3);
        for i=1:npatch
            if (mod(i,10000) == 0),
                fprintf('Extracting patch: %d / %d\n', i, npatch);
            end
            r = random('unid', CIFAR_DIM(1) - ws + 1);
            c = random('unid', CIFAR_DIM(2) - ws + 1);
            cpatch = reshape(xtr(mod(i-1,size(xtr,1))+1, :), CIFAR_DIM);
            cpatch = cpatch(r:r+ws-1,c:c+ws-1,:);
            patch(i,:) = cpatch(:)';
        end
        
        % normalize for contrast
        patch = bsxfun(@rdivide, bsxfun(@minus, patch, mean(patch,2)), sqrt(var(patch,[],2)+10));
        
        % whiten
        C = cov(patch);
        M = mean(patch);
        [V,D] = eig(C);
        P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';
        patch = bsxfun(@minus, patch, M) * P;
        
    else
        patch = [];
        M = [];
        P = [];
    end
    patch = single(patch);
    save(sprintf('patch/%s.mat', fname), 'patch', 'M', 'P', '-v7.3');
end

patch = double(patch);

return