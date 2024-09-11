function [bodyMovementData,rate,strokerisk,stateInBbed,inBedtime,rateMin,strokeriskMin,statesleep,timt,meansn,frame_img] = nczcode(frameData, tim)
    % 定义全局变量
    global G_INBED_TIM 
%     global G_OUTBED_TIM
%     global SW_INBED
%     global SW_OUTBED
%     global INBED_TIME %Bed leaving reminder record variable
    global realArr
%     global onBedArr
    global G_STATE_INBED
    global PRO_FRAME
    global ALL_DIFF_IMG
    global SEND_DATA_N  
    global STROKE_RISK
    global BREATH_RATE  
    global MIN_SEND_N
    global STROKE_RISK_MIN
    global BREATH_MIN
    global BATCH_FRA_MEAN
    global TWENTY_SEC       
    global STATE_SLEEP
    global FRAME_DIFF
    global ITER
    global VARN
%     global ONBED_SW
    global BREATH_SW
    global PRESSURE_MEAN
    global BODY_MOVE_NUM
%     global ALL_MOVE_LIST
%     global ALL_MOVE_NUM
    global still_or_life
    global inBedTimer
%     global previousFrames
%     global BREATH_COUNT
%     global TRIGGER_COUNT
%     global conditionTriggerHistory %still state trigger
%     global dominantFrequency
    global delay_count
    
      div_num = 6; %160大床整除系数
      frameData = round(frameData/div_num);
    % 调整数据顺序并去噪
    frameData_b = frameData; % copy of original data for breath
    frameData_b = reshape(frameData_b,32,32);
    frameData_b = frameData_b([1,2,3,4,5,6,7,8,9,10],[9:16,8:-1:1]);
    
    framedata_fall = frameData_b';
    
  
 
    frameData = reshape(frameData,32,32);
    frameData = frameData([1,2,3,4,5,6,7,8,9,10],[9:16,8:-1:1]);
    arrData = reshape(frameData,1,[]);
    frameData = Thrfilter(frameData,10,3);
 

    %% 初始化相关变量
    inBedFlag = 14 * VARN; %0621：2000/160
%     move_prop = VARN;
    bodyMovementData = (-1) * ones(24,1) * VARN;   
    strokerisk = (-1) * VARN;  
    strokeriskMin = (-1) * VARN;   
    rate = (-1) * VARN;
    rateMin = (-1) * VARN;
    stateInBbed = (-1) * VARN;    
    statesleep = (-1) * ones(48,1) * VARN; 
    inBedtime = (-1) * VARN;
    timt = (-1) * VARN;
    meansn = (-1) * VARN;
    frame_img = (-1) * ones(160,1) * VARN;
%     samplingRate = 12 * VARN;
%     alarm_switch = 0;
%     on_mat_flag = 0; %在垫标志位240703

    %% 计算矩阵的重心和压力
    frameData_160 = frameData;
    frameData_cy = frameData;
    [xSize, ySize] = size(frameData_cy);
    % 假设 frameData_cy 是你的矩阵
    frameData_cy(frameData_cy < 50) = 0;

    [~, yGrid] = meshgrid(1:xSize, 1:ySize);
    totalSum = sum(frameData_cy(:));
%     xCentroid = sum(xGrid(:) .* frameData_cy(:)) / totalSum;
    yCentroid = sum(yGrid(:) .* frameData_cy(:)) / totalSum;

    % 计算全床压力均值
    meanPressure = mean(frameData(:));
    % 设置上下限
    lowerBound = single(80);
    lowerBound1 = single(50);
    upperBound = single(255);
    frameData_single = single(frameData);
    % 调用函数
    [~, ~, countValue80,countValue60, ~] = analyzeMatrix(frameData_single, lowerBound,lowerBound1, upperBound);

    % 判断床标志位flag
    if yCentroid > 2 && yCentroid < 14 &&meanPressure > 18
        if  (countValue80>=3 || countValue60>10)%&& sumValue > 250 
            on_mat_flag = 1; % 在床
        else
            on_mat_flag = 0; % 离床
        end
    else
        on_mat_flag = 0; % 离床
    end

    %% 压力均值计算
    meanvar = mean2(frameData);  
    if meanvar > 0
        if PRESSURE_MEAN == 0 
            PRESSURE_MEAN = meanvar * VARN;
        else
            PRESSURE_MEAN = (PRESSURE_MEAN + meanvar * VARN) / 2; 
        end
    end

%     maxIndexNum = 1 * VARN;
%     onBedFlag = 1 * VARN;
    sequenceLen = 100;  
    onbedLen = 25;
%     maxIndexArr = zeros(1,sequenceLen) * VARN; 
    step = 1; % Compensation program single round run time

    %% 在床时间和状态判断
    if G_INBED_TIM == 0
        G_INBED_TIM = VARN;
        PRO_FRAME = frameData; 
        L2_arr = zeros(1,24);
    else
        SEND_DATA_N = SEND_DATA_N + step;                   
        TWENTY_SEC = TWENTY_SEC + step;
        MIN_SEND_N = MIN_SEND_N + step;
%         BREATH_COUNT = BREATH_COUNT + step;
%         TRIGGER_COUNT = TRIGGER_COUNT + step;
        BREATH_SW = BREATH_SW + 1;
        
        
%% 静物活物判断
        still_or_life_check = check_matrix(frameData_160, SEND_DATA_N);
        still_or_life = still_or_life_check*VARN;
        
        frameData_ncz = frameData;
        diffimgeD = sum(abs(frameData - PRO_FRAME));   
        STROKE_RISK = VARN - 2;
        % 脑卒中判断
        pro_frame = PRO_FRAME;
        [ncz_risk,L2_arr] = analyze_movement(pro_frame, frameData_ncz);

        if on_mat_flag>0%meanvar >= inBedFlag %&& still_or_life > 0
            % sosncz
            STROKE_RISK = VARN-1; %离床bug
            if (ncz_risk ~= 0) && still_or_life > 0
                STROKE_RISK = VARN * ncz_risk;
            end
        end

        PRO_FRAME = frameData_ncz;
        
        %statesleep 赋值
        FRAME_DIFF = FRAME_DIFF + diffimgeD';
        if TWENTY_SEC >= 240         
            STATE_SLEEP(ITER*4+1:(ITER+1)*4) = (FRAME_DIFF(1:4:end) + FRAME_DIFF(2:4:end) + FRAME_DIFF(3:4:end) + FRAME_DIFF(4:4:end)) / 240;
            ITER = ITER + 1; 
            FRAME_DIFF = VARN * zeros(16,1);
            TWENTY_SEC = VARN - 1;
            if ITER >= 3
                ITER = VARN - 1;
            end        
        end

        % 清醒入睡赋值 on_mat_flag
        G_STATE_INBED = VARN - 1;
        if meanvar >= inBedFlag %&& onBedFlag > 3.5        
            if on_mat_flag == 1
                G_STATE_INBED = VARN;
            end
            frameData_2 = frameData;
            frameData_2(frameData_2 < 1) = 0;
            meanvar2 = mean2(frameData_2(9:end,:));
            data = reshape(frameData',4,40);
            sumData = sum(data,1);
            data = sumData;
                  
            % 呼吸计算
            if BREATH_SW < sequenceLen                
                realArr(BREATH_SW, :) = data;  % 2023.5.31修改
                BATCH_FRA_MEAN(BREATH_SW) = meanvar2;
            else
                realArr(sequenceLen, :) = data;
                stdArr = std(realArr);
                [sortedA, sortedIndex] = sort(stdArr, 'descend');

                cutoff_lowpass = 0.6;
                sampling_rate = 13;
                nyquist_rate = sampling_rate / 2;
                cutoff_norm_lowpass = cutoff_lowpass / nyquist_rate;
                [b_lowpass, a_lowpass] = butter(4, cutoff_norm_lowpass, 'low');

                first_column1 = filtfilt(b_lowpass, a_lowpass, double(realArr(:,sortedIndex(1))));
                [peaks, locs] = findpeaks(first_column1, 'MinPeakDistance',10);
                avg_loc = mean(diff(locs));
                if length(locs) == 1
                    BREATH_RATE = BREATH_RATE + (600 * 0.8 / avg_loc - BREATH_RATE) / 10 * VARN;%原本乘以1.2
                    if BREATH_RATE<=10
                        BREATH_RATE = 10 * VARN;
                    end
                else
                    BREATH_RATE = BREATH_RATE + (600 * 0.8 / avg_loc - BREATH_RATE) / 10 * VARN;%原本乘以1.2
                end
                
                BREATH_SW = (sequenceLen - 12) * VARN;
                realArr(1:sequenceLen-12, :) = realArr(13:end,:);
            end
        else 
            BREATH_RATE = VARN - 1;
        end  
    end

    %% 体动赋值
    gain = 1.5;
    if SEND_DATA_N*gain >= 24  
        BODY_MOVE_NUM = BODY_MOVE_NUM+2;%时间计时临时变量
%         ALL_DIFF_IMG = L2_arr' * VARN;
        % 计算 L2_arr 的所有元素之和
        sum_L2 = sum(L2_arr, 'all');

        % 将和添加到 ALL_DIFF_IMG 的末尾
        ALL_DIFF_IMG = [ALL_DIFF_IMG(2:end); sum_L2]; % 此处是一个列向量

        % 应用 log2(1+x) 变换，减去 4，并将 32 以下的值归一化到 1
        bodyMovementData = log2(1 + ALL_DIFF_IMG) - 4;
        bodyMovementData = max(bodyMovementData, 1); % 将小于 1 的值（对应原始值小于 32）设为 1
        
%         bodyMovementData(bodyMovementData > 200) = 200;
%         bodyMovementData = round(bodyMovementData);%离散化
        bodyMovementData = min(bodyMovementData, 10); % 设置最高等级为 10上限等级
        % 体动强度映射表
        % 原始值范围    |  转换后等级  |  解释
        % 0-32         |     1       |  低强度体动
        % 33-63        |     2       |  轻度体动
        % 64-127       |     3       |  中度体动
        % 128-255      |     4       |  较强体动
        % 256-511      |     5       |  强烈体动
        % 512-1023     |     6       |  很强烈体动
        % 1024-2047    |     7       |  剧烈体动
        % 2048-4095    |     8       |  极剧烈体动
        % 4096-8191    |     9       |  超强体动
        % 8192+        |    10+      |  极端体动

        % 注：实际等级 = round(max(log2(1 + 原始值) - 4, 1))
        % 此映射为近似值，实际值可能略有不同
        
        
        
        
        
        if isempty(inBedTimer)
            inBedTimer = VARN - 1; % 确保持久变量已初始化
        end

        if (meanvar > inBedFlag) 
            inBedTimer = inBedTimer + 2;
            inBedtime = inBedTimer * VARN;  % 计算在床时间
        else
            inBedtime = VARN - 1;
            inBedTimer = VARN - 1;  % 重置计时器
        end

        stateInBbed = G_STATE_INBED;
        ratio = calculate_bed_status(framedata_fall);
        status = detectEdgeStatus(framedata_fall);
        %新在离床测试
        if ratio>=0.15
            stateInBbed = VARN;
        end
        if status>0 && ratio>=0.15
            stateInBbed = status * VARN;
        end
%         stateInBbed = VARN;%测试

        strokerisk = STROKE_RISK;
        STROKE_RISK = VARN - 1;

        if still_or_life > 0
            delay(20,1);
            if delay_count > 10
                rate = BREATH_RATE;
            else
                rate = VARN * 88;
            end
        else
            delay(20,0);
            if delay_count > 10
                rate = BREATH_RATE;
                
            elseif delay_count > 0
                rate = VARN * 88;
            else 
                rate = VARN - 1;  
                BREATH_RATE = VARN - 1;
            end
        end
%         rate = BODY_MOVE_NUM;%测试
        BREATH_MIN = BREATH_MIN + BREATH_RATE;
        SEND_DATA_N = VARN - 1;
        still_or_life = VARN - 1;
    end

    %% 数据处理和传输
    if MIN_SEND_N*gain >= 720
        statesleep = STATE_SLEEP;
        STROKE_RISK_MIN = max(STROKE_RISK_MIN, STROKE_RISK);
        strokeriskMin = STROKE_RISK_MIN;
        STROKE_RISK_MIN = VARN - 1;
        rateMin = BREATH_MIN / 30;
        BREATH_MIN = VARN - 1;
        
        %240731匹配睡眠算法包数据对接
        
        if 1 == on_mat_flag
            meansn = 25*VARN;
        else
            meansn = 5*VARN;
        end
        
        timt = tim;
        MIN_SEND_N = VARN - 1;
        PRESSURE_MEAN = VARN - 1;
        frame_img = reshape(frameData, 160, 1);
    end
end

%% 辅助函数

% 数据过滤函数
function Data = Thrfilter(Data, Thr, k)
    if sum(sum(Data)) > 0
        n_l = floor(k / 2);
        imgData = Data;
        imgData(imgData >= 1) = 1;
        [row, colum] = size(imgData);
        gimg = [zeros(n_l, colum); imgData; zeros(n_l, colum)];
        gimg = [zeros(size(gimg, 1), n_l), gimg, zeros(size(gimg, 1), n_l)];          
        kernel = ones(k);
        for i = n_l+1:n_l+row
            for j = n_l+1:n_l+colum
                if gimg(i, j) > 0
                    Block = gimg(i-n_l:i+n_l, j-n_l:j+n_l);
                    if (abs(sum(sum(Block .* kernel))) <= 1) && (Data(i-n_l, j-n_l) < Thr) 
                        Data(i-n_l, j-n_l) = 0;
                    end
                end
            end
        end
    end
end

% 频率分析函数
function [dominantFrequency] = analyzeSignalFrequency(signal, samplingRate)
    signalLength = length(signal);
    fftSignal = fft(signal);
    P2 = abs(fftSignal / signalLength);
    P1 = P2(1:floor(signalLength / 2) + 1);
    P1(2:end-1) = 2 * P1(2:end-1);
    f = samplingRate * (0:floor(signalLength / 2)) / signalLength;
    [pks, locs] = findpeaks(P1);
    [~, peakFreqIndex] = max(pks);
    dominantFrequency = f(locs(peakFreqIndex));
end

% 延迟函数
function delay(delay_time, delay_time_flag)
    global delay_count

    if delay_time_flag == 1
        if delay_count < delay_time
            delay_count = delay_count + 3;
        end
    else 
        if delay_count > 0
            delay_count = delay_count - 3;
        end
    end
end


%% 历史代码
function [strokerisk, arr1] = analyze_movement(PRO_FRAME, frameData)
    % 初始化输出变量
    strokerisk = 0;
%     arr = zeros(1, 24);
%     arr1 = zeros(1, 24);
    % 定义池化尺寸和其他参数
    pool_size = [2, 2];
    thresholds = [4, 8, 22]; % 分位数阈值
    win_size_stroke_initial = 96; % 第一次触发窗口大小
    win_size_stroke_subsequent = 48; % 后续触发窗口大小
    win_size_mov = 48; % 体动等级窗口
    proportion = 0.8; % 主要体动等级的比例

    % 将前一帧和当前帧数据转换为10x16矩阵
    prev_matrix = reshape(PRO_FRAME, 10, 16);
    curr_matrix = reshape(frameData, 10, 16);

    % 平均池化
    prev_pooled = pool_matrix(prev_matrix, pool_size);
    curr_pooled = pool_matrix(curr_matrix, pool_size);

    % 计算欧氏距离
    euclidean_distance = calculate_euclidean_distance(prev_pooled, curr_pooled);

    % 将欧氏距离分类为指定等级
    movement_level = classify_movement_level(euclidean_distance, thresholds);

    % 初始化报警变量和记录变量
    persistent movement_levels stroke_window_initial stroke_window_subsequent initial_check subsequent_check;
    persistent distance_arr; % 用于记录最近24帧的欧氏距离
    if isempty(movement_levels)
        movement_levels = zeros(1, win_size_mov);
        stroke_window_initial = zeros(1, win_size_stroke_initial);
        stroke_window_subsequent = zeros(1, win_size_stroke_subsequent);
        distance_arr = zeros(1, 24); % 初始化记录最近24帧的数组
        initial_check = false;
        subsequent_check = false;
    end

    % 更新体动等级窗口
    movement_levels = [movement_levels(2:end), movement_level];

    % 使用右侧法计算体动等级
    dominant_levels = dominant_movement_level_right(movement_levels, win_size_mov, proportion);


    % 更新记录最近24帧的欧氏距离数组
    distance_arr = [distance_arr(2:end), euclidean_distance];

    % 创建 arr 数组,将不在阈值区间内的值置零
    arr = distance_arr;
    arr1 = distance_arr;
    if euclidean_distance < thresholds(1) || euclidean_distance > thresholds(2)
        arr(end) = 0;
    end

    if ~initial_check
        % 更新初始脑卒中预警窗口
        stroke_window_initial = [stroke_window_initial(2:end), dominant_levels(end)];
        if sum(stroke_window_initial == 2) / win_size_stroke_initial >= 0.8
            strokerisk = 3; % 脑卒中预警
            initial_check = true;
            subsequent_check = true;
            stroke_window_subsequent = zeros(1, win_size_stroke_subsequent); % 重置后续窗口
        end
    elseif subsequent_check
        % 更新后续脑卒中预警窗口
        stroke_window_subsequent = [stroke_window_subsequent(2:end), dominant_levels(end)];
        if sum(stroke_window_subsequent == 2) / win_size_stroke_subsequent >= 0.8
            strokerisk = 3; % 脑卒中预警
            stroke_window_subsequent = zeros(1, win_size_stroke_subsequent); % 重置后续窗口
        else
            subsequent_check = false; % 重置检查
            initial_check = false; % 返回初始检查状态
            stroke_window_initial = zeros(1, win_size_stroke_initial); % 重置初始窗口
        end
    end
end

% 池化函数
function pooled_matrix = pool_matrix(matrix, pool_size)
    [rows, cols] = size(matrix);
    pooled_rows = floor(rows / pool_size(1));
    pooled_cols = floor(cols / pool_size(2));
    pooled_matrix = zeros(pooled_rows, pooled_cols);
    for i = 1:pooled_rows
        for j = 1:pooled_cols
            row_start = (i-1) * pool_size(1) + 1;
            row_end = i * pool_size(1);
            col_start = (j-1) * pool_size(2) + 1;
            col_end = j * pool_size(2);
            pooled_matrix(i, j) = mean(mean(matrix(row_start:row_end, col_start:col_end)));
        end
    end
end

% 欧氏距离计算函数
function distance = calculate_euclidean_distance(matrix1, matrix2)
    diff = matrix1 - matrix2;
    distance = sqrt(sum(diff(:) .^ 2));
end

% 将欧氏距离分类为指定等级的函数
function level = classify_movement_level(distance, thresholds)
    num_levels = length(thresholds) + 1;
    level = num_levels;
    for i = 1:length(thresholds)
        if distance <= thresholds(i)
            level = i;
            break;
        end
    end
end

% 使用右侧法计算窗口的主要体动等级函数
function dominant_levels = dominant_movement_level_right(levels, window_size, proportion)
    num_levels = length(levels) - window_size + 1;
    dominant_levels = zeros(1, num_levels);
    for i = 1:num_levels
        window = levels(i:i + window_size - 1);
        unique_levels = unique(window);
        level_counts = histcounts(window, [unique_levels - 0.5, max(unique_levels) + 0.5]);
        dominant_level = 0; % 默认等级为0
        for j = 1:length(unique_levels)
            if level_counts(j) / window_size >= proportion
                dominant_level = unique_levels(j);
                break;
            end
        end
        dominant_levels(i) = dominant_level;
    end
end

function [sumValue, meanValue, countValue80,countValue60, coordinates] = analyzeMatrix(frameData, lowerBound,lowerBound1, upperBound)
    % 确保输入是单精度浮点数
    frameData = single(reshape(frameData, 10, 16));
    lowerBound = single(lowerBound);
    lowerBound1 = single(lowerBound1);
    upperBound = single(upperBound);
    
    % 排除第一行
    frameData = frameData(:, 2:end-2);
    
    % 找到满足条件的元素
    [y, x] = find(frameData > lowerBound & frameData < upperBound);
    
    % 计算和、均值和个数
    values = frameData(frameData > lowerBound & frameData < upperBound);
    values1 = frameData(frameData > lowerBound1 & frameData < upperBound);
    
    % 初始化变量
    sumValue = single(0);
    meanValue = single(0);
    countValue80 = int32(0);
    countValue60 = int32(0);
    
    coordinates = zeros(0, 2, 'single'); % 初始化为单精度的 0x2 矩阵
    
    % 如果找到满足条件的元素
    if ~isempty(values)
        sumValue = sum(values);
        meanValue = mean(values);
        countValue80 = int32(numel(values));
        % 保存坐标（注意 MATLAB 的索引从 1 开始）
        coordinates = [single(x), single(y + 1)]; % y 坐标加 1
    end
%     values1
        if ~isempty(values1)
%         sumValue = sum(values);
        countValue60 = int32(numel(values1));
        end
end

function triggered = process_two_rows(rowData, SEND_DATA_N, pairIndex)
    % 声明全局变量
    global previousFramesArray BREATH_COUNTArray conditionTriggerHistoryArray TRIGGER_COUNTArray dominantFrequencyArray;

    % 初始化triggered
    triggered = false;
    rowData = rowData;%在160中，16是列
    
    % 计算方差
    varianceRow = var(rowData, 1, 'all');

    % 获取当前行对的变量
    previousFrames = previousFramesArray(:, pairIndex);
    BREATH_COUNT = BREATH_COUNTArray(pairIndex);
    conditionTriggerHistory = conditionTriggerHistoryArray(:, pairIndex);
    TRIGGER_COUNT = TRIGGER_COUNTArray(pairIndex);
    dominantFrequency = dominantFrequencyArray(pairIndex);

    % 更新previousFrames数组
    if BREATH_COUNT < 96
        BREATH_COUNT = BREATH_COUNT + 1;
        previousFrames(BREATH_COUNT) = varianceRow;
    else
        previousFrames(1:end-1) = previousFrames(2:end);
        previousFrames(end) = varianceRow;
    end

    if SEND_DATA_N*1.5 >= 24  % 每两秒进行一次FFT
        dominantFrequency = analyzeSignalFrequency(previousFrames, 8);
        
        % 检查当前运行条件是否触发
        conditionTriggered = (dominantFrequency <= 0.5) && (dominantFrequency >= 0.16);
        
        % 更新条件触发历史记录
        if TRIGGER_COUNT < 5
            TRIGGER_COUNT = TRIGGER_COUNT + 1;
            conditionTriggerHistory(TRIGGER_COUNT) = conditionTriggered;
        else
            conditionTriggerHistory(1:end-1) = conditionTriggerHistory(2:end);
            conditionTriggerHistory(end) = conditionTriggered;
        end

        % 如果条件触发历史记录满足条件，设置triggered为true
        if sum(conditionTriggerHistory) >= 4
            triggered = true;
        end
    end

    % 将更新后的变量存回全局数组
    previousFramesArray(:, pairIndex) = previousFrames;
    BREATH_COUNTArray(pairIndex) = BREATH_COUNT;
    conditionTriggerHistoryArray(:, pairIndex) = conditionTriggerHistory;
    TRIGGER_COUNTArray(pairIndex) = TRIGGER_COUNT;
    dominantFrequencyArray(pairIndex) = dominantFrequency;
end

%% 新呼吸检测函数
function still_or_life = check_matrix(frameData_1024, SEND_DATA_N)
    % 获取行数
    numRows = size(frameData_1024, 2);
    
    % 初始化still_or_life
    still_or_life = 0;
    
    % 初始化全局变量数组
    global previousFramesArray BREATH_COUNTArray conditionTriggerHistoryArray TRIGGER_COUNTArray dominantFrequencyArray;
    line_num = 1;
    numPairs = numRows / line_num;

    % 将每两行作为一组进行处理
    for i = 1:numPairs
        rowIndex = (i-1) * line_num + 1;
        rowData = frameData_1024(:,rowIndex:rowIndex+line_num-1);
        if process_two_rows(rowData, SEND_DATA_N, i)
            still_or_life = 1;
            break; % 触发后直接退出循环
        end
    end
end

function status = detectEdgeStatus(matrix)
    % 输入: matrix - 16x10的压力矩阵
    % 输出: status - 0(正常), 3(坠床), 4(坐床边)

    % 计算加权重心
    centroid = calculateWeightedCentroid(matrix);
    
    % 提取特征
    features = extract_features(matrix);
    
    % 计算概率
    probability = embeddedSystemLogic(features);
    
    % 判断状态
    if centroid(2) < 3.5 || centroid(2) > 7.5
        if probability > 0.5
            status = 3; % 坠床风险
        else
            status = 4; % 坐床边
        end
    elseif centroid(2) < 4 || centroid(2) >= 7
        if probability < 0.5
            status = 4; % 坐床边
        else
            status = 0; % 正常
        end
    else
        status = 0; % 正常
    end
end

function centroid = calculateWeightedCentroid(matrix)
    topN = 20;
    [sortedValues, sortedIndices] = sort(matrix(:), 'descend');
    topIndices = sortedIndices(1:topN);
    [rows, cols] = ind2sub(size(matrix), topIndices);
    topPoints = [rows, cols];
    pointValues = sortedValues(1:topN);
    totalWeight = sum(pointValues);
    centroid = sum(topPoints .* pointValues(:, [1 1]), 1) / totalWeight;
end

function features = extract_features(matrix)
    flat_matrix = matrix(:);
    
    % 计算矩阵元素的80分位数
    percentile_80 = mean(flat_matrix) + 0.8*std(flat_matrix);
    
    % 特征1: topN大且大于80分位数的点所在的唯一行的个数
    N = 32;
    [~, topN_indices] = sort(flat_matrix, 'descend');
    topN_indices = topN_indices(1:N);
    topN_values = flat_matrix(topN_indices);
    valid_topN_indices = topN_indices(topN_values > percentile_80);
    [valid_topN_rows, ~] = ind2sub(size(matrix), valid_topN_indices);
    unique_valid_topN_rows = unique(valid_topN_rows);
    
    features = length(unique_valid_topN_rows);
end

function probability = embeddedSystemLogic(features)
    coefficients = 5.21937961;
    intercept = -40.67752153154328;
    log_odds = features * coefficients + intercept;
    probability = 1 / (1 + exp(-log_odds));
end

% function features = extract_features(matrix)
%     [rows, cols] = size(matrix);
%     flat_matrix = matrix(:);
%     
%     % 特征1: top64大且大于15的点所在的唯一行的个数
%     [~, top64_indices] = sort(flat_matrix, 'descend');
%     top64_indices = top64_indices(1:min(32, length(top64_indices)));
%     top64_values = flat_matrix(top64_indices);
%     valid_top64_indices = top64_indices(top64_values > 35);
%     valid_top64_rows = mod(valid_top64_indices - 1, rows) + 1;
%     unique_valid_top64_rows = unique(valid_top64_rows);
%     
%     % 特征2: 大于50%阈值且大于15的点所在的唯一行的个数
%     min_val = min(flat_matrix);
%     max_val = max(flat_matrix);
%     threshold = min_val + 0.5 * (max_val - min_val);
%     
%     above_threshold_indices = find((flat_matrix >= threshold) & (flat_matrix > 35));
%     above_threshold_rows = mod(above_threshold_indices - 1, rows) + 1;
%     unique_above_threshold_rows = unique(above_threshold_rows);
%     
%     features = [length(unique_valid_top64_rows), length(unique_above_threshold_rows)];
% end
% 
% function probability = embeddedSystemLogic(features)
%     coefficients = [4.59846052, 1.42464485];
%     intercept = -43.540987268759594;
%     log_odds = features * coefficients' + intercept;
%     probability = 1 / (1 + exp(-log_odds));
% end

function ratio = calculate_bed_status(matrix)
    % 计算在床/离床状态的比例
    % 输入: matrix - 输入的矩阵
    % 输出: ratio - 在床/离床状态的比例

    % 将矩阵展平为一维数组，并按降序排序
    sorted_values = sort(double(matrix(:)), 'descend');

    % 计算前16个值的平均值
    top_16_median = mean(sorted_values(1:16));

    % 计算前48个值的平均值
    top_32_median = mean(sorted_values(1:48));

    % 计算调和平均值
    if top_16_median + top_32_median > 0
        harmonic_mean = 2 * (top_16_median * top_32_median) / (top_16_median + top_32_median);
    else
        harmonic_mean = 0;
    end

    % 计算比例
    ratio = harmonic_mean / 255;
end