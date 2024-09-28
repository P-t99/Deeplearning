/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: predictTree.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

#ifndef PREDICTTREE_H
#define PREDICTTREE_H

/* Include Files */
#include "predictTree_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void predictTree(const double XTest[160], categorical *YPred);

void predictTree_init(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for predictTree.h
 *
 * [EOF]
 */
