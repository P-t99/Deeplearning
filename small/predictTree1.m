% 假设 XTest 是一个包含待预测数据的矩阵
XTest = [0,0,0,0,1,9,1,0,0,0,0,0,0,5,2,3,0,0,0,0,0,0,0,5,26,68,51,3,0,0,0,0,0,4,44,34,33,40,1,0,0,0,0,3,30,25,2,1,0,0,0,0,0,2,21,32,4,0,0,0,0,0,0,2,36,21,12,0,0,0,0,0,0,3,36,25,5,0,0,0,0,0,0,46,33,44,5,0,0,0,0,0,0,54,61,38,21,1,0,0,0,0,0,2,16,31,37,1,0,0,0,0,0,0,1,17,9,0,0,0,0,0,0,0,1,10,2,0,0,0,0,0,0,6,33,4,0,0,0,0,1,3,8,8,1,0,0,0,0,0,3,6,3,1,0,0,0,0,1,1];  % 示例输入矩阵

% 调用 predictTree 函数进行预测
YPred = predictTree(XTest);

% 输出预测结果
disp(YPred);
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