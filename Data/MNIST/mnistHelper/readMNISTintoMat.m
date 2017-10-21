% Read the MNIST Dataset into MATLAB .mat files.
% 
% The two helper functions are found at 
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
clear; close;
train_images_file = '../train-images.idx3-ubyte';
train_labels_file = '../train-labels.idx1-ubyte';
test_images_file = '../t10k-images.idx3-ubyte';
test_labels_file = '../t10k-labels.idx1-ubyte';
trainX = loadMNISTdata(train_images_file); % each column as a training sample
trainy = loadMNISTdata(train_labels_file);
testX = loadMNISTdata(test_images_file);
testy = loadMNISTdata(test_labels_file);
save('MINIST.mat', 'trainX', 'trainy', 'testX', 'testy');
