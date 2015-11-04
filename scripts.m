acc = {};
fname = {};
if ~exist('optgpu', 'var'),
    optgpu = 0;
end

% translation
acc{end+1} = demo_cifar10(optgpu, 8, 1600, 'trans', 2, 0, 2, 0.1, 3, 0.01);
fname{end+1} = 'translation';

% rotation
acc{end+1} = demo_cifar10(optgpu, 6, 1600, 'rot', 1, 5, 1, 0.1, 3, 0.01);
fname{end+1} = 'rotation';

% scale
acc{end+1} = demo_cifar10(optgpu, 8, 1600, 'scale', 2, 0, 2, 0.1, 3, 0.01);
fname{end+1} = 'scale';


% print results
for i = 1:length(acc),
    fprintf('%s: %g\n', fname{i}, acc{i});
end


