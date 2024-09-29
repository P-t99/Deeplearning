function YPred = predictTree(XTest)
    % 加载预训练的决策树模型
    persistent treeModel
    if isempty(treeModel)
        loadedData = coder.load('finalPrunedTreeModel.mat');
        treeModel = loadedData.finalPrunedTree;
    end
    
    % 归一化输入数据
    XTest_normalized = normalizeData(XTest);
    
    % 使用归一化后的数据进行预测
    YPred = predict(treeModel, XTest_normalized);
end

function X_normalized = normalizeData(X)
    X = single(X);  % 转换为单精度
    lower_percentile = prctile(X, 2);
    upper_percentile = prctile(X, 98);
    
    % 归一化到 0-255 范围并取整
    normalized = (X - lower_percentile) / (upper_percentile - lower_percentile) * 255;
    X_normalized = round(max(0, min(255, normalized)));
end