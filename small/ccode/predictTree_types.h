/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: predictTree_types.h
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

#ifndef PREDICTTREE_TYPES_H
#define PREDICTTREE_TYPES_H

/* Include Files */
#include "rtwtypes.h"

/* Type Definitions */
#ifndef struct_emxArray_char_T_1x11
#define struct_emxArray_char_T_1x11
struct emxArray_char_T_1x11 {
  char data[11];
  int size[2];
};
#endif /* struct_emxArray_char_T_1x11 */
#ifndef typedef_emxArray_char_T_1x11
#define typedef_emxArray_char_T_1x11
typedef struct emxArray_char_T_1x11 emxArray_char_T_1x11;
#endif /* typedef_emxArray_char_T_1x11 */

#ifndef typedef_cell_wrap_1
#define typedef_cell_wrap_1
typedef struct {
  emxArray_char_T_1x11 f1;
} cell_wrap_1;
#endif /* typedef_cell_wrap_1 */

#ifndef typedef_categorical
#define typedef_categorical
typedef struct {
  unsigned char codes;
  cell_wrap_1 categoryNames[3];
} categorical;
#endif /* typedef_categorical */

#endif
/*
 * File trailer for predictTree_types.h
 *
 * [EOF]
 */
