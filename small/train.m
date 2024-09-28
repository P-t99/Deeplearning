% 导入CSV文件
data = readtable('train.csv');

% 将字符串矩阵转换为数值矩阵
X = cellfun(@str2num, data.data, 'UniformOutput', false);
X = cell2mat(X);

% 准备标签
Y = categorical(data.position);

% 对每个1x160矩阵单独进行最大最小值归一化
X_normalized = zeros(size(X));
for i = 1:size(X, 1)
    X_min = min(X(i, :));
    X_max = max(X(i, :));
    X_normalized(i, :) = (X(i, :) - X_min) / (X_max - X_min);
end

% 将数据分为训练集和测试集
[trainInd,testInd] = dividerand(size(X_normalized,1), 0.8, 0.2);
XTrain = X_normalized(trainInd,:);
YTrain = Y(trainInd);
XTest = X_normalized(testInd,:);
YTest = Y(testInd);

% 创建决策树分类器
tree = fitctree(XTrain, YTrain);

% 使用交叉验证来评估模型
cv = crossval(tree);
loss = kfoldLoss(cv);

disp(['交叉验证误差: ' num2str(loss)]);

% 在测试集上进行预测
YPred = predict(tree, XTest);

% 计算准确率
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['测试集准确率: ' num2str(accuracy)]);

% 绘制混淆矩阵
confusionchart(YTest, YPred);

% 保存模型
save('trainedTreeModel.mat', 'tree');

disp('模型已保存为 trainedTreeModel.mat');