/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: nczallcode_types.h
 *
 * Code generated for Simulink model 'nczallcode'.
 *
 * Model version                  : 7.17
 * Simulink Coder version         : 24.1 (R2024a) 19-Nov-2023
 * C/C++ source code generated on : Wed Oct  9 14:09:47 2024
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: NXP->Cortex-M4
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef nczallcode_types_h_
#define nczallcode_types_h_
#include "rtwtypes.h"

/* Custom Type definition for MATLAB Function: '<Root>/MATLAB Function' */
#ifndef struct_tag_LxXjg0auJ7ZK291Djn5EvG
#define struct_tag_LxXjg0auJ7ZK291Djn5EvG

struct tag_LxXjg0auJ7ZK291Djn5EvG
{
  real_T CutPredictorIndex[227];
  real_T Children[454];
  real_T CutPoint[227];
  real_T PruneList[227];
  boolean_T NanCutPoints[227];
  boolean_T InfCutPoints[227];
  int32_T ClassNamesLength[3];
  real_T Prior[3];
  real_T Cost[9];
  int32_T CharClassNamesLength[3];
  real_T ClassProbability[681];
};

#endif                                 /* struct_tag_LxXjg0auJ7ZK291Djn5EvG */

#ifndef typedef_ClassificationTree_nczallcode_T
#define typedef_ClassificationTree_nczallcode_T

typedef struct tag_LxXjg0auJ7ZK291Djn5EvG ClassificationTree_nczallcode_T;

#endif                             /* typedef_ClassificationTree_nczallcode_T */

#ifndef struct_emxArray_char_T_1x11
#define struct_emxArray_char_T_1x11

struct emxArray_char_T_1x11
{
  char_T data[11];
  int32_T size[2];
};

#endif                                 /* struct_emxArray_char_T_1x11 */

#ifndef typedef_emxArray_char_T_1x11_nczallco_T
#define typedef_emxArray_char_T_1x11_nczallco_T

typedef struct emxArray_char_T_1x11 emxArray_char_T_1x11_nczallco_T;

#endif                             /* typedef_emxArray_char_T_1x11_nczallco_T */

/* Custom Type definition for MATLAB Function: '<Root>/MATLAB Function' */
#ifndef struct_tag_QjBXI9rtubOTIDmB1WWuBH
#define struct_tag_QjBXI9rtubOTIDmB1WWuBH

struct tag_QjBXI9rtubOTIDmB1WWuBH
{
  emxArray_char_T_1x11_nczallco_T f1;
};

#endif                                 /* struct_tag_QjBXI9rtubOTIDmB1WWuBH */

#ifndef typedef_cell_wrap_1_nczallcode_T
#define typedef_cell_wrap_1_nczallcode_T

typedef struct tag_QjBXI9rtubOTIDmB1WWuBH cell_wrap_1_nczallcode_T;

#endif                                 /* typedef_cell_wrap_1_nczallcode_T */

/* Forward declaration for rtModel */
typedef struct tag_RTM_nczallcode_T RT_MODEL_nczallcode_T;

#endif                                 /* nczallcode_types_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
