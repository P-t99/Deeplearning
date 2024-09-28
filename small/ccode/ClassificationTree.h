/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ClassificationTree.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

#ifndef CLASSIFICATIONTREE_H
#define CLASSIFICATIONTREE_H

/* Include Files */
#include "predictTree_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
unsigned char ClassificationTree_predict(
    const double obj_CutPredictorIndex[227], const double obj_Children[454],
    const double obj_CutPoint[227], const double obj_PruneList[227],
    const bool obj_NanCutPoints[227], const double obj_Prior[3],
    const double obj_Cost[9], const double obj_ClassProbability[681],
    const double Xin[160], cell_wrap_1 labels_categoryNames[3]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ClassificationTree.h
 *
 * [EOF]
 */
