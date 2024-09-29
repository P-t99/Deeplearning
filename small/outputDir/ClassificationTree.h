/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ClassificationTree.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

#ifndef CLASSIFICATIONTREE_H
#define CLASSIFICATIONTREE_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
float ClassificationTree_predict(
    const float obj_CutPredictorIndex[65], const float obj_Children[130],
    const float obj_CutPoint[65], const float obj_PruneList[65],
    const boolean_T obj_NanCutPoints[65], const float obj_Cost[9],
    const float obj_ClassProbability[195], const float Xin[160]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ClassificationTree.h
 *
 * [EOF]
 */
