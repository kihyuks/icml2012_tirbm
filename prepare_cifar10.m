CIFAR_DIR = 'cifar-10-batches-mat';
CIFAR_DIM = [32 32 3];

if ~exist(CIFAR_DIR, 'dir'),
    system('wget http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz');
    system('tar -zxvf cifar-10-matlab.tar.gz');
    system('rm cifar-10-matlab.tar.gz');
end
