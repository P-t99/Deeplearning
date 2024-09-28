% 加载原始数据（假设您的原始数据文件名为'train.csv'）
data = readtable('train.csv');

% 将字符串矩阵转换为数值矩阵
X = cellfun(@str2num, data.data, 'UniformOutput', false);
X = cell2mat(X);

% 准备标签
Y = categorical(data.position);

% 随机抽取10个样本的索引
numSamples = 10;
totalSamples = size(X, 1);
randomIndices = randperm(totalSamples, numSamples);

% 初始化正确预测计数器
correctPredictions = 0;

% 测试随机抽取的10个样本
disp('测试随机抽取的10个样本：');
for i = 1:numSamples
    sampleIndex = randomIndices(i);
    XTest = X(sampleIndex, :);
    
    % 对测试样本进行最大最小值归一化
    X_min = min(XTest);
    X_max = max(XTest);
    XTest_normalized = (XTest - X_min) / (X_max - X_min);
    
    % 使用predictTree函数进行预测
    YPred = predictTree(XTest_normalized);
    actualLabel = Y(sampleIndex);
    
    disp(['样本 ', num2str(sampleIndex)]);
    disp(['预测结果: ', char(YPred)]);
    disp(['实际标签: ', char(actualLabel)]);
    
    % 检查预测是否正确
    if YPred == actualLabel
        disp('预测正确！');
        correctPredictions = correctPredictions + 1;
    else
        disp('预测错误。');
    end
    disp('---');
end

% 计算并显示总正确率
accuracy = correctPredictions / numSamples;
disp(['总正确率: ', num2str(accuracy * 100), '%']);