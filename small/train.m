% 数据准备
data = readtable('train.csv');
X = cellfun(@(x) str2num(x), data.data, 'UniformOutput', false);
X = cat(3, X{:});  % 将所有矩阵堆叠成一个三维数组

% 打印X的维度
disp(['X的维度: ', num2str(size(X))]);

% % 选取前三个矩阵，绘制热力图
% figure;
% for i = 1:3
%     subplot(1,3,i);
%     imagesc(reshape(X(:,:,i), 10, 16));  % 从三维数组中取矩阵并reshape
%     colorbar;
%     title(['第', num2str(i), '个矩阵']);
%     axis equal;
%     axis tight;
% end

Y = data.position;
% 设置全局随机种子
rng(42, 'twister');  % 42 可以替换为任何整数

% 数据归一化
X_normalized = normalizeData(X);

% 数据分割
[trainInd, testInd] = dividerand(size(X_normalized,3), 0.8, 0.2);
XTrain = X_normalized(:,:,trainInd);
YTrain = Y(trainInd);
XTest = X_normalized(:,:,testInd);
YTest = Y(testInd);

% 重塑 XTrain 和 XTest 为二维矩阵
XTrain = reshape(XTrain, [], size(XTrain, 3))';
XTest = reshape(XTest, [], size(XTest, 3))';

% 确保 YTrain 和 YTest 也是正确的数据类型
YTrain = single(YTrain);
YTest = single(YTest);

% 定义超参数优化的目标函数
cvErr = @(x) kfoldLoss(crossval(fitctree(XTrain, YTrain, ...
    'MinLeafSize', x.minLeaf, 'MaxNumSplits', x.maxNumSplits), 'KFold', 5));

% 定义超参数的搜索范围
minLeaf = optimizableVariable('minLeaf', [1, 20], 'Type', 'integer');
maxNumSplits = optimizableVariable('maxNumSplits', [1, 50], 'Type', 'integer');

% 执行贝叶斯优化
results = bayesopt(cvErr, [minLeaf, maxNumSplits], ...
    'Verbose', 1, ...
    'UseParallel', true, ...
    'MaxObjectiveEvaluations', 30, ...
    'PlotFcn', {@plotObjectiveModel, @plotMinObjective}, ...
    'AcquisitionFunctionName', 'expected-improvement-plus');

% 使用优化后的参数训练最终模型
finalTree = fitctree(XTrain, YTrain, ...
    'MinLeafSize', results.XAtMinObjective.minLeaf, ...
    'MaxNumSplits', results.XAtMinObjective.maxNumSplits);

% 保存模型和测试数据
save('finalTreeModel.mat', 'finalTree', 'XTest', 'YTest');
disp('模型和测试数据已保存为 finalTreeModel.mat');

% 显示原始树的信息
disp(['原始树节点数: ' num2str(finalTree.NumNodes)]);

% 评估原始模型
YPred = predict(finalTree, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['原始树测试集准确率: ' num2str(accuracy)]);

% 添加新的归一化函数
function X_normalized = normalizeData(X)
    [~, ~, n] = size(X);
    X_normalized = zeros(size(X), 'single');  % 使用单精度浮点数
    
    for i = 1:n
        current_matrix = single(X(:,:,i));  % 转换为单精度
        lower_percentile = prctile(current_matrix(:), 2);
        upper_percentile = prctile(current_matrix(:), 98);
        
        % 归一化到 0-255 范围并取整
        normalized = (current_matrix - lower_percentile) / (upper_percentile - lower_percentile) * 255;
        X_normalized(:,:,i) = single(round(max(0, min(255, normalized))));
    end
end