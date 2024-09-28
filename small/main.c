#define _CRT_SECURE_NO_WARNINGS
#include "predictTree.h"  // 这是生成的头文件名，可能会有所不同
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main() {
    double input[160] = {0};
    char line[1000];
    int count = 0;

    while (1) {  // 添加无限循环
        printf("Please enter an array of 160 numbers in the format [1,2,3,...,160]:\n");
        printf("Enter 'q' to quit the program.\n");  // 添加退出提示

        if (fgets(line, sizeof(line), stdin) != NULL) {
            if (line[0] == 'q' || line[0] == 'Q') {  // 检查是否要退出
                printf("Program exited.\n");
                break;
            }

            char *start = strchr(line, '[');
            char *end = strrchr(line, ']');
            if (start && end && start < end) {
                *end = '\0';  // 移除结尾的 ']'
                start++;  // 跳过开头的 '['
                char *token = strtok(start, ",");
                while (token != NULL && count < 160) {
                    input[count++] = atof(token);
                    token = strtok(NULL, ",");
                }
            }
        }

        if (count != 160) {
            printf("Error: The number of input values is not 160. Actual input: %d\n", count);
            continue;  // 继续下一次循环，而不是退出程序
        }

        // 打印输入的数组以验证
        printf("The array you entered is:\n[");
        for (int i = 0; i < 160; i++) {
            printf("%.1f", input[i]);
            if (i < 159) printf(", ");
        }
        printf("]\n");

        // 初始化预测结果变量
        categorical YPred;

        // 初始化预测树（如果需要）
        predictTree_init();

        // 调用预测函数
        predictTree(input, &YPred);

        // 打印结果
        int pre = YPred.codes;
        printf("Prediction result: %d\n\n", pre);

        // 重置计数器和输入数组，为下一次循环做准备
        count = 0;
        memset(input, 0, sizeof(input));
    }

    return 0;
}