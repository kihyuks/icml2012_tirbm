function [acc, weight, params] = demo_cifar10...
    (optgpu, rs, numhid, txtype, numtx, numrot, grid, pbias, plambda, eta_sigma, split)

startup;

dataset = 'cifar10';
intype = 'real';
sptype = 'exact';
epsilon = 0.005;
batchSize = 200;
if ~exist('split', 'var'),
    split = 'split'; end

% hyper parameters
if ~exist('optgpu', 'var'),
    optgpu = 0; end
if ~exist('rs', 'var'),
    rs = 6; end
if ~exist('numhid', 'var'),
    numhid = 200; end
if ~exist('grid', 'var'),
    grid = 1; end
if ~exist('numtx', 'var'),
    numtx = 1; end
if ~exist('numrot', 'var'),
    numrot = 1; end
if ~exist('txtype', 'var'),
    txtype = 'trans'; end
if ~exist('pbias', 'var'),
    pbias = 0.1; end
if ~exist('plambda', 'var'),
    plambda = 3; end
if ~exist('eta_sigma', 'var'),
    eta_sigma = 0.01; end


% -----------------------------------------------------------------------
%                                       hyper parameters for dataset
% -----------------------------------------------------------------------

params.dataset = dataset;
params.rs = rs;
params.numhid = numhid;
params.optgpu = optgpu;
params.grid = grid;
params.txtype = txtype;
params.numtx = numtx;
params.numrot = numrot;
params.dataset = dataset;
params.pbias = pbias;
params.plambda = plambda;
params.intype = intype;
params.sptype = sptype;
params.savepath = savedir;

% other hyper parameters
params.maxiter = 150;
params.batchsize = batchSize;
params.epsilon = epsilon;
params.eta_sigma = eta_sigma;
params.l2reg = 1e-4;
params.epsdecay = 0.01;
params.kcd = 1;
params.numch = 3;


% -----------------------------------------------------------------------
%                                   load training and testing images
% -----------------------------------------------------------------------

npatch = 400000;
[xtrain, ytrain, patches, M, P] = load_patches(npatch, params.rs);


% -----------------------------------------------------------------------
%                                   generate transformation matrices
% -----------------------------------------------------------------------

params.ws = params.rs - (params.numtx-1)*params.grid;
params.scales = params.ws:params.grid:params.rs;
params.rSize = params.rs^2*params.numch;
params.numvis = params.ws^2*params.numch;

Tlist = get_txmat(params.txtype, params.rs, params.ws, params.grid, params.numrot, params.numch);
params.numtx = length(Tlist);


% -----------------------------------------------------------------------
%                                                         train TIRBM
% -----------------------------------------------------------------------

% filename to save
if strcmp(params.txtype, 'rot'),
    fname = sprintf('trbm_%s_w%d_b%02d_%s_nrot%d_pb%g_pl%g', ...
        params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.pbias, params.plambda);
elseif strcmp(params.txtype, 'trans') || strcmp(params.txtype, 'scale'),
    fname = sprintf('trbm_%s_w%d_b%02d_%s_ntx%d_gr%d_pb%g_pl%g', ...
        params.dataset, params.ws, params.numhid, params.txtype, params.numtx, params.grid, params.pbias, params.plambda);
end
params.fname  = sprintf('%s/%s', params.savepath, fname);

try
    load([params.fname '_iter_' num2str(params.maxiter) '.mat'], 'weight', 'params');
catch
    [weight, params] = tirbm_train(patches', params, Tlist);
end
clear patches;


% -----------------------------------------------------------------------
%                                 Evaluate classification performance
% -----------------------------------------------------------------------

% load CIFAR test dataset
fprintf('Loading test dataset...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);
xtest = double(f1.data);
ytest = double(f1.labels) + 1;
clear f1;

% extract training and testing features
fprintf('Extracting features...\n');
xtrainC = tirbm_inference(xtrain, params.rs, weight, params, Tlist, M, P, optgpu, split);
xtestC = tirbm_inference(xtest, params.rs, weight, params, Tlist, M, P, optgpu, split);

% standardize dataset
xtrainC_mean = mean(xtrainC);
xtrainC_sd = sqrt(var(xtrainC)+0.01);
xtrainCs = bsxfun(@rdivide, bsxfun(@minus, xtrainC, xtrainC_mean), xtrainC_sd);
xtrainCs = [xtrainCs, ones(size(xtrainCs, 1), 1)];
xtestCs = bsxfun(@rdivide, bsxfun(@minus, xtestC, xtrainC_mean), xtrainC_sd);
xtestCs = [xtestCs, ones(size(xtestCs, 1), 1)];

C = 30;
theta = train_svm(xtrainCs, ytrain, C);

[~, labels] = max(xtestCs*theta, [], 2);
acc = 100 * (1 - sum(labels ~= ytest) / length(ytest));

fprintf('Ts: %g (optC:%g, %s_iter_%d)\n', acc, C, fname, params.maxiter);

return;
