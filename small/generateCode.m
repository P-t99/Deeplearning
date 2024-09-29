% 定义输入参数的大小和类型
    XTest = coder.typeof(double(zeros(1, 160))); % 保持原来的1x160双精度数组
    
    % 配置代码生成器
    cfg = coder.config('lib');
    cfg.GenerateReport = true;
    cfg.ReportPotentialDifferences = false;
    
    % 针对嵌入式设备的额外配置
    cfg.HardwareImplementation.ProdHWDeviceType = 'ARM Compatible->ARM Cortex';
    cfg.HardwareImplementation.TargetHWDeviceType = 'ARM Compatible->ARM Cortex';
    cfg.EnableMemcpy = false;
    
    % 使用新的动态内存分配选项
    cfg.EnableDynamicMemoryAllocation = 'Off';
    
    % 设置目标语言为C
    cfg.TargetLang = 'C';
    
    % 设置只生成代码，不编译
    cfg.GenCodeOnly = true;
    
    % 定义输出目录和文件名
    outputDir = 'generated_code';
    outputFileName = 'predictTree';
    
    % 生成C代码
    try
        codegen -config cfg predictTree -args {XTest} -report -o outputFileName -d outputDir
        disp('C代码生成成功完成。');
        disp(['生成的文件位于: ', fullfile(pwd, outputDir)]);
    catch ME
        fprintf('代码生成失败：%s\n', ME.message);
        fprintf('请查看报告以获取更多详细信息：codegen/lib/predictTree/html/report.mldatx\n');
    end
    
    % 修改MATLAB文件的添加
    coder.updateBuildInfo('addMATLABFile', 'finalPrunedTreeModel.mat');