

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
    if centroid(2) < 3.5 || centroid(2) > 6.5
        if probability > 0.5
            status = 3; % 坠床风险
        else
            status = 4; % 坐床边
        end
    elseif centroid(2) < 4 || centroid(2) >= 6
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


        % 更新记录最近24帧的欧氏距离数组
        distance_arr = [distance_arr(2:end), euclidean_distance];

        % 创建 arr 数组,将不在阈值区间内的值置零
        arr = distance_arr;
        arr1 = distance_arr;
        if euclidean_distance < thresholds(1) || euclidean_distance > thresholds(2)
            arr(end) = 0;
        end


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