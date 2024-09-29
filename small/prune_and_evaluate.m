% 加载原始树模型和测试数据
load('finalTreeModel.mat');

% 计算最佳剪枝级别
[~,~,~,bestLevel] = cvLoss(finalTree, 'SubTrees', 'all');

% 尝试不同的剪枝级别
for additionalPruning = 0:20
    prunedTree = prune(finalTree, 'Level', bestLevel + additionalPruning);
    
    % 评估模型
    YPred = predict(prunedTree, XTest);
    accuracy = sum(YPred == YTest) / numel(YTest);
    
    disp(['剪枝级别 ' num2str(bestLevel + additionalPruning) ...
          '，准确率: ' num2str(accuracy) ...
          '，节点数: ' num2str(prunedTree.NumNodes)]);
end

% 选择一个你认为最佳的剪枝级别
bestAdditionalPruning = 15;  % 这里你可以根据上面的结果选择一个最佳值
finalPrunedTree = prune(finalTree, 'Level',bestAdditionalPruning);

% 保存最终剪枝后的模型
save('finalPrunedTreeModel.mat', 'finalPrunedTree');
disp('最终剪枝后的模型和归一化参数已保存为 finalPrunedTreeModel.mat');