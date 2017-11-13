% the MINIST data is originally 784-dimensional
% apply PCA to reduce the dimension to 2d and 3d
% then visualize the reduced data

close all; clear;

%% load data: trainX, trainy, testX, testy (each column is a data sample)
load('../Data/MNIST/MNIST.mat')

%% apply PCA to trainX and the get the principle directions
[P2d, ~, trainX2d] = PCA(trainX, 2);
[P3d, ~, trainX3d] = PCA(trainX, 3);


%% visualize the projected data
% define 10 colors for the 10 classes
colors = [1, 0, 0;
          0, 1, 0;
          0, 0, 1;
          1, 1, 0;
          1, 0, 1;
          0, 1, 1;
          0, 0, 0;
          1, 0.5, 0.8;
          0, 0.7, 0.5;
          0.2, 0.3, 0.7];
          
% 2d
% trainX2d = P2d' * trainX;
figure;
hold on;
for ii = 0:9
   filter = trainy==ii;
   scatter(trainX2d(1, filter), trainX2d(2, filter), 10, colors(ii+1,:), 'filled'); %marker size 10
end
hold off;
xlabel('PC1');
ylabel('PC2');
legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');

% 3d
% trainX3d = P3d' * trainX;
figure;
hold on;
for ii = 0:9
   filter = trainy==ii;
   scatter3(trainX3d(1, filter), trainX3d(2, filter), trainX3d(3, filter), 10, colors(ii+1,:), 'filled'); %marker size 10
end
hold off;
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
legend('0', '1', '2', '3', '4', '5', '6', '7', '8', '9');
grid on;


%% visualize the eigenvector (principle axes)
% reshape the principel axis vector into a 28-by-28 matrix, then display the matrix as an image

% 2d
figure;
for ii = 1:2
    subplot(1, 2, ii);
    image = reshape(P2d(:, ii), [28, 28]); 
    imshow(image); % the data in image will be automatically normalized into range [0, 1]
    title(sprintf('PC%d', ii));
end

% 3d
figure;
for ii = 1:3
    subplot(1, 3, ii);
    image = reshape(P3d(:, ii), [28, 28]); 
    % image = imcomplement(image); % reverse black and white
    imshow(image); % the data in image will be automatically normalized into range [0, 1]
    title(sprintf('PC%d', ii));
end




