/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: introsort.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

/* Include Files */
#include "introsort.h"
#include "insertionsort.h"
#include "predictTree_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const cell_wrap_1 cmp_workspace_c[3]
 *                int x[3]
 * Return Type  : void
 */
void introsort(const cell_wrap_1 cmp_workspace_c[3], int x[3])
{
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  insertionsort(x, cmp_workspace_c);
}

/*
 * File trailer for introsort.c
 *
 * [EOF]
 */
