function YPred = predictTree(XTest)
    % 加载预训练的决策树模型
    persistent treeModel
    if isempty(treeModel)
        treeModel = loadCompactModel('trainedTreeModel.mat');
    end
    
    % 对输入数据进行归一化
    X_min = min(XTest, [], 2);
    X_max = max(XTest, [], 2);
    XTest_normalized = (XTest - X_min) ./ (X_max - X_min + eps);
    
    % 预测
    YPred = predict(treeModel, XTest_normalized);
end

function model = loadCompactModel(filename)
    % 加载紧凑模型
    s = coder.load(filename);
    model = s.tree;  % 将 s.treeModel 改为 s.tree
end