#define _CRT_SECURE_NO_WARNINGS
#include "predictTree.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    double input[160] = { 0 };
    char line[1000];
    int count = 0;

    while (1) {
        printf("请输入160个数字的数组，格式为 [1,2,3,...,160]：\n");
        printf("输入 'q' 退出程序。\n");

        if (fgets(line, sizeof(line), stdin) != NULL) {
            if (line[0] == 'q' || line[0] == 'Q') {
                printf("程序已退出。\n");
                break;
            }

            char* start = strchr(line, '[');
            char* end = strrchr(line, ']');
            if (start && end && start < end) {
                *end = '\0';
                start++;
                char* token = strtok(start, ",");
                while (token != NULL && count < 160) {
                    input[count++] = atof(token);
                    token = strtok(NULL, ",");
                }
            }
        }

        if (count != 160) {
            printf("错误：输入值的数量不是160。实际输入：%d\n", count);
            count = 0;
            memset(input, 0, sizeof(input));
            continue;
        }

        printf("您输入的数组是：\n[");
        for (int i = 0; i < 160; i++) {
            printf("%.1f", input[i]);
            if (i < 159) printf(", ");
        }
        printf("]\n");

        // 初始化预测树（如果需要）
        predictTree_initialize();

        // 调用预测函数
        float YPred = predictTree(input);

        // 打印结果
        printf("预测结果：%.0f\n\n", YPred);

        // 重置计数器和输入数组，为下一次循环做准备
        count = 0;
        memset(input, 0, sizeof(input));
    }

    // 终止预测树
    predictTree_terminate();

    return 0;
}