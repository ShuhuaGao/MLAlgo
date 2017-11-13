
% Apply LDA to reduce data dimensionality from 784 to 2, 3 and 9. Visualize distribution of the
% data with dimensionality of 2 and 3 respectively (similar to PCA). Report the classification accuracy 
% for data with dimensions of 2, 3 and 9 respectively, based on nearest neighbor classifier.
% Test the maximal dimensionality that data can be projected to via LDA. Explain the reasons.
clear; close all;
%% load data: trainX, trainy, testX, testy (each column is a data sample)
load('../../Data/MNIST/MNIST.mat');
[m, N] = size(trainX); % m attributes and N samples in total
n = NaN(10, 1); 
% count the size of each class (0 - 9)
for c = 0: 9
    n(c + 1) = sum(trainy == c);
end
p = n ./ N; % prior probability of each class (frequency)

%% FLD: scatter matrix
% 1. sample covariance matrix of each class
S = NaN(m, m, 10);
muc = NaN(m, 10);
for c = 0:9
    Xc = trainX(:, trainy == c);
    muc(:, c+1) = mean(Xc, 2);
    S(:, :, c + 1) = 1 / n(c + 1) * (Xc - muc(:, c+1)) * (Xc - muc(:, c+1))'; 
end
% 2. with-in class scatter: weighted sum of the class-specific sample covariance
Sw = zeros(m, m);
for c = 0:9
    Sw = Sw + p(c+1) * S(:, :, c+1);
end
% 3. between-class scatter
mu = mean(trainX, 2); % mean of all samples in all classes
Sb = (muc - mu) * diag(p) * (muc - mu)';
% % (3). between-class scatter computed by Sb = St - Sw
% St = 1 / N * (trainX - mu) * (trainX - mu)'; % sample covariance matrix of the whole data 
% Sbt = St - Sw;
% % here we can show verify that St = Sb + Sw

%% solve the generalized eigen value problem
fprintf('rank of Sw = %d\n', rank(Sw));
[U, D, V] = svd(Sb);
r = rank(Sb); % only pick positive eigenvalues
Ur = U(:, 1:r);
d = diag(D);
dr = d(1:r);
Dr = diag(dr);
Z = Ur * diag(1./sqrt(dr));
[P, Dw] = svd(Z'*Sw*Z);
dw = diag(Dw);
Q = Z * P * diag(1./sqrt(dw));
% flip
Lambda = diag(1 ./ fliplr(dw));
Q = fliplr(Q);


%% visualize LDA dimension reduction into 2d and 3d
colors = {'dark blue', 'almond', 'black', 'tea rose (rose)', 'blue gray', 
          'bright maroon', 'bubbles', 'gold (metallic)', 'pearl aqua', 'violet (web)'};
trainXr = cell(1, 2); 
for ii = [2, 3]
    W = Q(:, 1:ii);
    trainXr{ii - 1} = W' * trainX; % apply the projection
end

figure('Name', '2d');
colormap(summer(10));
hold on;
legends = cell(1, 10);
for c = 0:9
    X = trainXr{1}(:, trainy == c);
    scatter(X(1, :), X(2, :), 4, cmu.colors(colors{c+1}), 'filled');
    legends{c+1} = sprintf('class %d', c);
end
legend(legends{:});
xlabel('LDA1');
ylabel('LDA2');
hold off;

figure('Name', '3d');
colormap(summer(10));
hold on;
legends = cell(1, 10);
for c = 0:9
    X = trainXr{2}(:, trainy == c);
    scatter3(X(1, :), X(2, :), X(3,:), 4, cmu.colors(colors{c+1}), 'filled');
    legends{c+1} = sprintf('class %d', c);
end
legend(legends{:});
xlabel('LDA1');
ylabel('LDA2');
zlabel('LDA3');
grid on;
hold off;

%%  nearest neighbor
dim = [2, 3, 9];
accuracy = NaN(1, 3);
testp = NaN(size(testy));
for ii = 1:3
    W = Q(:, 1:dim(ii));
    trainXr = W' * trainX;
    testXr = W' * testX;
    for jj = 1: length(testy)
        x = testXr(:, jj);
        % find the nearest neighbor of x in the reduced trainX
        vn = sum((trainXr - x).^2);
        [~, index] = min(vn);
        testp(jj) = trainy(index);
    end
    fprintf('accurary for d = %d: %.4f\n', dim(ii), sum(testp == testy) / length(testp));
end







