/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ClassificationTree.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

/* Include Files */
#include "ClassificationTree.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const float obj_CutPredictorIndex[65]
 *                const float obj_Children[130]
 *                const float obj_CutPoint[65]
 *                const float obj_PruneList[65]
 *                const boolean_T obj_NanCutPoints[65]
 *                const float obj_Cost[9]
 *                const float obj_ClassProbability[195]
 *                const float Xin[160]
 * Return Type  : float
 */
float ClassificationTree_predict(
    const float obj_CutPredictorIndex[65], const float obj_Children[130],
    const float obj_CutPoint[65], const float obj_PruneList[65],
    const boolean_T obj_NanCutPoints[65], const float obj_Cost[9],
    const float obj_ClassProbability[195], const float Xin[160])
{
  float a__6[3];
  float ex;
  float f;
  int i;
  int k;
  int m;
  boolean_T exitg1;
  m = 0;
  exitg1 = false;
  while (!(exitg1 || (obj_PruneList[m] <= 0.0F))) {
    f = Xin[(int)obj_CutPredictorIndex[m] - 1];
    if (rtIsNaNF(f) || obj_NanCutPoints[m]) {
      exitg1 = true;
    } else if (f < obj_CutPoint[m]) {
      m = (int)obj_Children[m << 1] - 1;
    } else {
      m = (int)obj_Children[(m << 1) + 1] - 1;
    }
  }
  for (i = 0; i < 3; i++) {
    a__6[i] = (obj_ClassProbability[m] * obj_Cost[3 * i] +
               obj_ClassProbability[m + 65] * obj_Cost[3 * i + 1]) +
              obj_ClassProbability[m + 130] * obj_Cost[3 * i + 2];
  }
  if (!rtIsNaNF(a__6[0])) {
    m = 1;
  } else {
    m = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 4)) {
      if (!rtIsNaNF(a__6[k - 1])) {
        m = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (m == 0) {
    m = 1;
  } else {
    ex = a__6[m - 1];
    i = m + 1;
    for (k = i; k < 4; k++) {
      f = a__6[k - 1];
      if (ex > f) {
        ex = f;
        m = k;
      }
    }
  }
  return (float)m;
}

/*
 * File trailer for ClassificationTree.c
 *
 * [EOF]
 */
