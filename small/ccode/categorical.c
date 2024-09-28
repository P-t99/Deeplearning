/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: categorical.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

/* Include Files */
#include "categorical.h"
#include "introsort.h"
#include "predictTree_types.h"
#include "rt_nonfinite.h"
#include "strtrim.h"

/* Function Definitions */
/*
 * Arguments    : const cell_wrap_1 inputData
 *                const cell_wrap_1 varargin_1[3]
 *                cell_wrap_1 b_categoryNames[3]
 * Return Type  : unsigned char
 */
unsigned char categorical_categorical(const cell_wrap_1 inputData,
                                      const cell_wrap_1 varargin_1[3],
                                      cell_wrap_1 b_categoryNames[3])
{
  cell_wrap_1 this_workspace_c[3];
  cell_wrap_1 inData;
  int unusedExpr[3];
  int b_i;
  int i;
  int loop_ub;
  bool exitg1;
  strtrim(inputData.f1.data, inputData.f1.size, inData.f1.data, inData.f1.size);
  for (i = 0; i < 3; i++) {
    strtrim(varargin_1[i].f1.data, varargin_1[i].f1.size,
            this_workspace_c[i].f1.data, this_workspace_c[i].f1.size);
  }
  introsort(this_workspace_c, unusedExpr);
  b_categoryNames[0].f1.size[0] = 1;
  loop_ub = this_workspace_c[0].f1.size[1];
  b_categoryNames[0].f1.size[1] = this_workspace_c[0].f1.size[1];
  for (b_i = 0; b_i < loop_ub; b_i++) {
    b_categoryNames[0].f1.data[b_i] = this_workspace_c[0].f1.data[b_i];
  }
  b_categoryNames[1].f1.size[0] = 1;
  loop_ub = this_workspace_c[1].f1.size[1];
  b_categoryNames[1].f1.size[1] = this_workspace_c[1].f1.size[1];
  for (b_i = 0; b_i < loop_ub; b_i++) {
    b_categoryNames[1].f1.data[b_i] = this_workspace_c[1].f1.data[b_i];
  }
  b_categoryNames[2].f1.size[0] = 1;
  loop_ub = this_workspace_c[2].f1.size[1];
  b_categoryNames[2].f1.size[1] = this_workspace_c[2].f1.size[1];
  for (b_i = 0; b_i < loop_ub; b_i++) {
    b_categoryNames[2].f1.data[b_i] = this_workspace_c[2].f1.data[b_i];
  }
  b_i = 0;
  i = 0;
  exitg1 = false;
  while ((!exitg1) && (i < 3)) {
    bool b;
    bool b_bool;
    b_bool = false;
    loop_ub = this_workspace_c[i].f1.size[1];
    b = (inData.f1.size[1] == 0);
    if (b && (loop_ub == 0)) {
      b_bool = true;
    } else if (inData.f1.size[1] == loop_ub) {
      int kstr;
      kstr = 0;
      int exitg2;
      do {
        exitg2 = 0;
        if (kstr <= loop_ub - 1) {
          if (inData.f1.data[0] != this_workspace_c[i].f1.data[0]) {
            exitg2 = 1;
          } else {
            kstr = 1;
          }
        } else {
          b_bool = true;
          exitg2 = 1;
        }
      } while (exitg2 == 0);
    }
    if (b_bool) {
      b_i = i + 1;
      exitg1 = true;
    } else {
      i++;
    }
  }
  return (unsigned char)b_i;
}

/*
 * File trailer for categorical.c
 *
 * [EOF]
 */
