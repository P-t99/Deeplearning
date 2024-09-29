/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_predictTree_api.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

#ifndef _CODER_PREDICTTREE_API_H
#define _CODER_PREDICTTREE_API_H

/* Include Files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
real32_T predictTree(real_T XTest[160]);

void predictTree_api(const mxArray *prhs, const mxArray **plhs);

void predictTree_atexit(void);

void predictTree_initialize(void);

void predictTree_terminate(void);

void predictTree_xil_shutdown(void);

void predictTree_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for _coder_predictTree_api.h
 *
 * [EOF]
 */
