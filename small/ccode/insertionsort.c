/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: insertionsort.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

/* Include Files */
#include "insertionsort.h"
#include "predictTree_types.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : int x[3]
 *                const cell_wrap_1 cmp_workspace_c[3]
 * Return Type  : void
 */
void insertionsort(int x[3], const cell_wrap_1 cmp_workspace_c[3])
{
  int k;
  for (k = 0; k < 2; k++) {
    int idx;
    int xc;
    bool exitg1;
    xc = x[k + 1] - 1;
    idx = k;
    exitg1 = false;
    while ((!exitg1) && (idx + 1 >= 1)) {
      int b_k;
      int b_n_tmp;
      int j;
      int n;
      int n_tmp;
      bool varargout_1;
      j = x[idx];
      n_tmp = cmp_workspace_c[xc].f1.size[1];
      b_n_tmp = cmp_workspace_c[x[idx] - 1].f1.size[1];
      n = (int)fmin(n_tmp, b_n_tmp);
      varargout_1 = (n_tmp < b_n_tmp);
      b_k = 0;
      int exitg2;
      do {
        exitg2 = 0;
        if (b_k <= n - 1) {
          if (cmp_workspace_c[xc].f1.data[0] !=
              cmp_workspace_c[x[idx] - 1].f1.data[0]) {
            varargout_1 = (cmp_workspace_c[xc].f1.data[0] <
                           cmp_workspace_c[x[idx] - 1].f1.data[0]);
            exitg2 = 1;
          } else {
            b_k = 1;
          }
        } else {
          if (n_tmp == b_n_tmp) {
            varargout_1 = (xc + 1 < j);
          }
          exitg2 = 1;
        }
      } while (exitg2 == 0);
      if (varargout_1) {
        x[idx + 1] = x[idx];
        idx--;
      } else {
        exitg1 = true;
      }
    }
    x[idx + 1] = xc + 1;
  }
}

/*
 * File trailer for insertionsort.c
 *
 * [EOF]
 */
