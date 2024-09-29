/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: predictTree_initialize.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

/* Include Files */
#include "predictTree_initialize.h"
#include "predictTree.h"
#include "predictTree_data.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void predictTree_initialize(void)
{
  rt_InitInfAndNaN();
  predictTree_init();
  isInitialized_outputFileName = true;
}

/*
 * File trailer for predictTree_initialize.c
 *
 * [EOF]
 */
