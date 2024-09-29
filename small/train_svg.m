% 导入CSV文件
data = readtable('train.csv');

% 将字符串矩阵转换为数值矩阵
X = cellfun(@str2num, data.data, 'UniformOutput', false);
X = cell2mat(X);

% 准备标签
Y = categorical(data.position);

% 直接使用原始数据
[trainInd,testInd] = dividerand(size(X,1), 0.8, 0.2);
XTrain = X(trainInd,:);
YTrain = Y(trainInd);
XTest = X(testInd,:);
YTest = Y(testInd);

% 创建线性核SVM分类器
svm = fitcecoc(XTrain, YTrain, 'Learners', templateSVM('KernelFunction', 'linear'));

% 使用交叉验证来评估模型
cv = crossval(svm);
loss = kfoldLoss(cv);

disp(['交叉验证误差: ' num2str(loss)]);

% 在原始测试集上进行预测
YPred = predict(svm, XTest);

% 计算原始测试集的准确率
originalAccuracy = sum(YPred == YTest) / numel(YTest);
disp(['原始测试集准确率: ' num2str(originalAccuracy)]);

% 将测试数据放大10倍
XTestScaled = XTest * 10;

% 在放大后的测试集上进行预测
YPredScaled = predict(svm, XTestScaled);

% 计算放大后测试集的准确率
scaledAccuracy = sum(YPredScaled == YTest) / numel(YTest);
disp(['放大后测试集准确率: ' num2str(scaledAccuracy)]);

% 比较两个准确率
if abs(originalAccuracy - scaledAccuracy) < 1e-6
    disp('SVM对数据等比放大不敏感');
else
    disp('SVM对数据等比放大敏感');
end

% 绘制混淆矩阵
confusionchart(YTest, YPred);

% 保存模型
save('trainedSVMModel.mat', 'svm');

disp('模型已保存为 trainedSVMModel.mat');