% 加载模型
loadedData = load('trainedTreeModel.mat', 'tree');
tree = loadedData.tree;

% 使用whos查看模型大小
whos tree

% 获取树的属性
numNodes = size(tree.CutPredictor, 1);
disp(['节点数量: ', num2str(numNodes)]);

% 估算模型大小（粗略估计）
estimatedSize = numNodes * (8 + 8 + 4); % 每个节点：分割值(double) + 子节点索引(double) + 预测变量索引(int32)
disp(['估计模型大小（字节）: ', num2str(estimatedSize)]);

% 生成C代码
codegen predictTree -args {coder.typeof(zeros(1,160))} -report -config:mex

% 分析生成的C代码
fid = fopen('codegen/mex/predictTree/predictTree_data.c', 'r');
if fid == -1
    error('无法打开文件。请确保代码生成成功并且文件路径正确。');
end
content = fread(fid, '*char')';
fclose(fid);

% 查找静态数组声明
arrays = regexp(content, 'static const \w+ \w+\[\d+\]', 'match');
totalSize = 0;

for i = 1:length(arrays)
    tokens = regexp(arrays{i}, 'static const (\w+) \w+\[(\d+)\]', 'tokens');
    type = tokens{1}{1};
    size = str2double(tokens{1}{2});
    switch type
        case 'double'
            totalSize = totalSize + size * 8;
        case 'int32_T'
            totalSize = totalSize + size * 4;
        otherwise
            disp(['未知类型: ', type]);
    end
end

disp(['C代码中静态数组的总大小（字节）: ', num2str(totalSize)]);