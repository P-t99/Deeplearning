/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: nczallcode.c
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

#include "nczallcode.h"
#include "rtwtypes.h"
#include "nczallcode_types.h"
#include <math.h>
#include "rt_nonfinite.h"
#include <string.h>

/* Block states (default storage) */
DW_nczallcode_T nczallcode_DW;

/* External inputs (root inport signals with default storage) */
ExtU_nczallcode_T nczallcode_U;

/* External outputs (root outports fed by signals with default storage) */
ExtY_nczallcode_T nczallcode_Y;

/* Real-time model */
static RT_MODEL_nczallcode_T nczallcode_M_;
RT_MODEL_nczallcode_T *const nczallcode_M = &nczallcode_M_;

/* Forward declaration for local functions */
static void nczallcode_strtrim(const char_T x_data[], const int32_T x_size[2],
  char_T y_data[], int32_T y_size[2]);
static void nczallcode_insertionsort(int32_T x[3], const
  cell_wrap_1_nczallcode_T cmp_workspace_c[3]);
static void nczallcode_introsort(int32_T x[3], const cell_wrap_1_nczallcode_T
  cmp_workspace_c[3]);
static uint8_T nczallc_categorical_categorical(const cell_wrap_1_nczallcode_T
  *inputData, const cell_wrap_1_nczallcode_T varargin_1[3]);
static uint8_T ncza_ClassificationTree_predict(const real_T
  obj_CutPredictorIndex[227], const real_T obj_Children[454], const real_T
  obj_CutPoint[227], const real_T obj_PruneList[227], const boolean_T
  obj_NanCutPoints[227], const real_T obj_Prior[3], const real_T obj_Cost[9],
  const real_T obj_ClassProbability[681], const real32_T Xin[160]);

/* Function for MATLAB Function: '<Root>/MATLAB Function' */
static void nczallcode_strtrim(const char_T x_data[], const int32_T x_size[2],
  char_T y_data[], int32_T y_size[2])
{
  int32_T b_j1;
  int32_T i;
  int32_T j2;
  static const boolean_T d[128] = { false, false, false, false, false, false,
    false, false, false, true, true, true, true, true, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    true, true, true, true, true, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false };

  b_j1 = 0;
  while ((b_j1 + 1 <= x_size[1]) && d[(int32_T)x_data[b_j1]]) {
    b_j1++;
  }

  j2 = x_size[1];
  while ((j2 > 0) && d[(int32_T)x_data[j2 - 1]]) {
    j2--;
  }

  if (b_j1 + 1 > j2) {
    b_j1 = 0;
    j2 = 0;
  }

  y_size[0] = 1;
  j2 -= b_j1;
  y_size[1] = j2;
  for (i = 0; i < j2; i++) {
    y_data[i] = x_data[b_j1 + i];
  }
}

/* Function for MATLAB Function: '<Root>/MATLAB Function' */
static void nczallcode_insertionsort(int32_T x[3], const
  cell_wrap_1_nczallcode_T cmp_workspace_c[3])
{
  int32_T k;
  for (k = 0; k < 2; k++) {
    int32_T idx;
    int32_T xc;
    boolean_T exitg1;
    xc = x[k + 1] - 1;
    idx = k;
    exitg1 = false;
    while ((!exitg1) && (idx + 1 >= 1)) {
      int32_T b_k;
      int32_T j;
      int32_T n;
      int32_T n_tmp;
      int32_T n_tmp_0;
      boolean_T varargout_1;
      j = x[idx];
      n_tmp = cmp_workspace_c[xc].f1.size[1];
      n_tmp_0 = cmp_workspace_c[x[idx] - 1].f1.size[1];
      n = (int32_T)fmin(n_tmp, n_tmp_0);
      varargout_1 = (n_tmp < n_tmp_0);
      b_k = 0;
      int32_T exitg2;
      do {
        exitg2 = 0;
        if (b_k <= n - 1) {
          char_T tmp;
          char_T tmp_0;
          tmp = cmp_workspace_c[xc].f1.data[0];
          tmp_0 = cmp_workspace_c[x[idx] - 1].f1.data[0];
          if (tmp_0 != tmp) {
            varargout_1 = (tmp < tmp_0);
            exitg2 = 1;
          } else {
            b_k++;
          }
        } else {
          if (n_tmp_0 == n_tmp) {
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

/* Function for MATLAB Function: '<Root>/MATLAB Function' */
static void nczallcode_introsort(int32_T x[3], const cell_wrap_1_nczallcode_T
  cmp_workspace_c[3])
{
  nczallcode_insertionsort(x, cmp_workspace_c);
}

/* Function for MATLAB Function: '<Root>/MATLAB Function' */
static uint8_T nczallc_categorical_categorical(const cell_wrap_1_nczallcode_T
  *inputData, const cell_wrap_1_nczallcode_T varargin_1[3])
{
  cell_wrap_1_nczallcode_T a[3];
  cell_wrap_1_nczallcode_T b_a;
  int32_T f[3];
  int32_T b_i;
  int32_T exitg2;
  int32_T i;
  int32_T kstr;
  int32_T nb_tmp;
  uint8_T b_codes;
  boolean_T b_bool;
  boolean_T d;
  boolean_T exitg1;
  nczallcode_strtrim(inputData->f1.data, inputData->f1.size, b_a.f1.data,
                     b_a.f1.size);
  for (i = 0; i < 3; i++) {
    nczallcode_strtrim(varargin_1[i].f1.data, varargin_1[i].f1.size, a[i].
                       f1.data, a[i].f1.size);
    f[i] = i + 1;
  }

  nczallcode_introsort(f, a);
  i = 0;
  b_i = 0;
  exitg1 = false;
  while ((!exitg1) && (b_i < 3)) {
    b_bool = false;
    nb_tmp = a[b_i].f1.size[1];
    d = (b_a.f1.size[1] == 0);
    if (d && (nb_tmp == 0)) {
      b_bool = true;
    } else if (b_a.f1.size[1] == nb_tmp) {
      kstr = 0;
      do {
        exitg2 = 0;
        if (kstr <= nb_tmp - 1) {
          if (b_a.f1.data[0] != a[b_i].f1.data[0]) {
            exitg2 = 1;
          } else {
            kstr++;
          }
        } else {
          b_bool = true;
          exitg2 = 1;
        }
      } while (exitg2 == 0);
    }

    if (b_bool) {
      i = b_i + 1;
      exitg1 = true;
    } else {
      b_i++;
    }
  }

  b_codes = (uint8_T)i;
  return b_codes;
}

/* Function for MATLAB Function: '<Root>/MATLAB Function' */
static uint8_T ncza_ClassificationTree_predict(const real_T
  obj_CutPredictorIndex[227], const real_T obj_Children[454], const real_T
  obj_CutPoint[227], const real_T obj_PruneList[227], const boolean_T
  obj_NanCutPoints[227], const real_T obj_Prior[3], const real_T obj_Cost[9],
  const real_T obj_ClassProbability[681], const real32_T Xin[160])
{
  cell_wrap_1_nczallcode_T ain[4];
  cell_wrap_1_nczallcode_T b_ain[4];
  cell_wrap_1_nczallcode_T b[3];
  cell_wrap_1_nczallcode_T a;
  cell_wrap_1_nczallcode_T b_labels;
  cell_wrap_1_nczallcode_T d;
  cell_wrap_1_nczallcode_T e;
  cell_wrap_1_nczallcode_T f;
  real_T x[160];
  real_T b_x[3];
  real_T b_x_0;
  real_T ex;
  int32_T b_k;
  int32_T c_k;
  int32_T idx;
  int32_T iindx;
  int32_T m;
  boolean_T c_x[3];
  boolean_T y;
  static const char_T h[11] = { '<', 'u', 'n', 'd', 'e', 'f', 'i', 'n', 'e', 'd',
    '>' };

  static const char_T i[3] = { '1', '2', '3' };

  boolean_T exitg1;
  for (idx = 0; idx < 160; idx++) {
    x[idx] = Xin[idx];
  }

  m = 0;
  exitg1 = false;
  while (!(exitg1 || (obj_PruneList[m] <= 0.0))) {
    ex = x[(int32_T)obj_CutPredictorIndex[m] - 1];
    if (rtIsNaN(ex) || obj_NanCutPoints[m]) {
      exitg1 = true;
    } else if (ex < obj_CutPoint[m]) {
      m = (int32_T)obj_Children[m << 1] - 1;
    } else {
      m = (int32_T)obj_Children[(m << 1) + 1] - 1;
    }
  }

  for (idx = 0; idx < 3; idx++) {
    b_x[idx] = (obj_Cost[3 * idx + 1] * obj_ClassProbability[m + 227] +
                obj_Cost[3 * idx] * obj_ClassProbability[m]) + obj_Cost[3 * idx
      + 2] * obj_ClassProbability[m + 454];
  }

  ex = obj_Prior[0];
  iindx = -1;
  if (obj_Prior[0] < obj_Prior[1]) {
    ex = obj_Prior[1];
    iindx = 0;
  }

  if (ex < obj_Prior[2]) {
    iindx = 1;
  }

  if (!rtIsNaN(b_x[0])) {
    idx = 1;
  } else {
    idx = 0;
    b_k = 2;
    exitg1 = false;
    while ((!exitg1) && (b_k < 4)) {
      if (!rtIsNaN(b_x[b_k - 1])) {
        idx = b_k;
        exitg1 = true;
      } else {
        b_k++;
      }
    }
  }

  if (idx == 0) {
    b_k = 1;
  } else {
    ex = b_x[idx - 1];
    b_k = idx;
    for (c_k = idx + 1; c_k < 4; c_k++) {
      b_x_0 = b_x[c_k - 1];
      if (ex > b_x_0) {
        ex = b_x_0;
        b_k = c_k;
      }
    }
  }

  c_x[0] = rtIsNaN(obj_ClassProbability[m]);
  c_x[1] = rtIsNaN(obj_ClassProbability[m + 227]);
  c_x[2] = rtIsNaN(obj_ClassProbability[m + 454]);
  y = true;
  m = 0;
  exitg1 = false;
  while ((!exitg1) && (m < 3)) {
    if (!c_x[m]) {
      y = false;
      exitg1 = true;
    } else {
      m++;
    }
  }

  b_ain[0].f1.size[0] = 1;
  b_ain[0].f1.size[1] = 11;
  for (idx = 0; idx < 11; idx++) {
    b_ain[0].f1.data[idx] = h[idx];
  }

  b_ain[1].f1.size[0] = 1;
  b_ain[1].f1.size[1] = 1;
  b_ain[1].f1.data[0] = '1';
  b_ain[2].f1.size[0] = 1;
  b_ain[2].f1.size[1] = 1;
  b_ain[2].f1.data[0] = '2';
  b_ain[3].f1.size[0] = 1;
  b_ain[3].f1.size[1] = 1;
  b_ain[3].f1.data[0] = '3';
  b_labels = b_ain[iindx + 2];
  if (!y) {
    b_labels.f1.size[0] = 1;
    b_labels.f1.size[1] = 1;
    b_labels.f1.data[0] = i[b_k - 1];
  }

  nczallcode_strtrim(b_labels.f1.data, b_labels.f1.size, a.f1.data, a.f1.size);
  iindx = 0;
  y = false;
  if ((a.f1.size[1] == 1) && (a.f1.data[0] == '1')) {
    y = true;
  }

  if (y) {
    iindx = 1;
  } else {
    y = false;
    if ((a.f1.size[1] == 1) && (a.f1.data[0] == '2')) {
      y = true;
    }

    if (y) {
      iindx = 2;
    } else {
      y = false;
      if ((a.f1.size[1] == 1) && (a.f1.data[0] == '3')) {
        y = true;
      }

      if (y) {
        iindx = 3;
      }
    }
  }

  ain[0].f1.size[0] = 1;
  ain[0].f1.size[1] = 11;
  ain[1].f1.size[0] = 1;
  ain[1].f1.size[1] = 1;
  ain[1].f1.data[0] = '1';
  ain[2].f1.size[0] = 1;
  ain[2].f1.size[1] = 1;
  ain[2].f1.data[0] = '2';
  ain[3].f1.size[0] = 1;
  ain[3].f1.size[1] = 1;
  ain[3].f1.data[0] = '3';
  for (idx = 0; idx < 11; idx++) {
    ain[0].f1.data[idx] = h[idx];
  }

  b_labels = ain[iindx];
  d.f1.size[0] = 1;
  d.f1.size[1] = 1;
  d.f1.data[0] = '1';
  e.f1.size[0] = 1;
  e.f1.size[1] = 1;
  e.f1.data[0] = '2';
  f.f1.size[0] = 1;
  f.f1.size[1] = 1;
  f.f1.data[0] = '3';
  b[0] = d;
  b[1] = e;
  b[2] = f;
  return nczallc_categorical_categorical(&b_labels, b);
}

/* Model step function */
void nczallcode_step(void)
{
  int32_T b_k;
  int32_T k;
  real32_T frameData[160];
  real32_T tmp[160];
  real32_T X_max;
  real32_T X_min;
  real32_T frameData_0;
  uint8_T a_codes;
  boolean_T tmp_0;
  static const uint8_T b[227] = { 103U, 66U, 147U, 77U, 118U, 17U, 65U, 106U,
    115U, 5U, 91U, 39U, 8U, 5U, 83U, 7U, 139U, 94U, 76U, 64U, 44U, 0U, 1U, 131U,
    114U, 0U, 0U, 0U, 6U, 0U, 0U, 76U, 21U, 75U, 3U, 84U, 16U, 4U, 24U, 13U,
    117U, 0U, 16U, 0U, 0U, 70U, 0U, 0U, 1U, 0U, 0U, 5U, 36U, 0U, 0U, 34U, 80U,
    0U, 0U, 108U, 76U, 24U, 0U, 0U, 4U, 0U, 0U, 15U, 0U, 24U, 0U, 0U, 0U, 82U,
    7U, 0U, 0U, 0U, 0U, 0U, 59U, 23U, 121U, 14U, 0U, 35U, 0U, 98U, 27U, 0U, 0U,
    0U, 0U, 0U, 0U, 1U, 65U, 26U, 2U, 0U, 3U, 0U, 0U, 59U, 4U, 0U, 3U, 0U, 0U,
    0U, 0U, 39U, 26U, 4U, 3U, 0U, 0U, 0U, 18U, 51U, 47U, 0U, 0U, 0U, 0U, 128U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 14U, 0U, 0U, 0U, 0U, 0U, 38U, 154U, 0U, 0U,
    0U, 159U, 4U, 0U, 0U, 11U, 0U, 143U, 76U, 88U, 4U, 0U, 0U, 27U, 0U, 148U, 3U,
    3U, 0U, 32U, 0U, 7U, 0U, 0U, 18U, 18U, 0U, 0U, 0U, 0U, 0U, 91U, 0U, 0U, 0U,
    1U, 0U, 49U, 0U, 34U, 0U, 0U, 0U, 45U, 0U, 76U, 3U, 9U, 70U, 13U, 15U, 3U,
    0U, 0U, 0U, 93U, 0U, 77U, 0U, 0U, 0U, 0U, 0U, 70U, 0U, 23U, 75U, 26U, 0U, 0U,
    2U, 0U, 0U, 144U, 3U, 0U, 0U, 0U, 7U, 0U, 0U, 0U, 0U };

  static const uint8_T c[454] = { 2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U,
    13U, 14U, 15U, 16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U, 24U, 25U, 26U, 27U,
    28U, 29U, 30U, 31U, 32U, 33U, 34U, 35U, 36U, 37U, 38U, 39U, 40U, 41U, 42U,
    43U, 0U, 0U, 44U, 45U, 46U, 47U, 48U, 49U, 0U, 0U, 0U, 0U, 0U, 0U, 50U, 51U,
    0U, 0U, 0U, 0U, 52U, 53U, 54U, 55U, 56U, 57U, 58U, 59U, 60U, 61U, 62U, 63U,
    64U, 65U, 66U, 67U, 68U, 69U, 70U, 71U, 0U, 0U, 72U, 73U, 0U, 0U, 0U, 0U,
    74U, 75U, 0U, 0U, 0U, 0U, 76U, 77U, 0U, 0U, 0U, 0U, 78U, 79U, 80U, 81U, 0U,
    0U, 0U, 0U, 82U, 83U, 84U, 85U, 0U, 0U, 0U, 0U, 86U, 87U, 88U, 89U, 90U, 91U,
    0U, 0U, 0U, 0U, 92U, 93U, 0U, 0U, 0U, 0U, 94U, 95U, 0U, 0U, 96U, 97U, 0U, 0U,
    0U, 0U, 0U, 0U, 98U, 99U, 100U, 101U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    102U, 103U, 104U, 105U, 106U, 107U, 108U, 109U, 0U, 0U, 110U, 111U, 0U, 0U,
    112U, 113U, 114U, 115U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 116U,
    117U, 118U, 119U, 120U, 121U, 122U, 123U, 0U, 0U, 124U, 125U, 0U, 0U, 0U, 0U,
    126U, 127U, 128U, 129U, 0U, 0U, 130U, 131U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    132U, 133U, 134U, 135U, 136U, 137U, 138U, 139U, 0U, 0U, 0U, 0U, 0U, 0U, 140U,
    141U, 142U, 143U, 144U, 145U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 146U, 147U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 148U, 149U, 0U,
    0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 150U, 151U, 152U, 153U, 0U, 0U, 0U, 0U,
    0U, 0U, 154U, 155U, 156U, 157U, 0U, 0U, 0U, 0U, 158U, 159U, 0U, 0U, 160U,
    161U, 162U, 163U, 164U, 165U, 166U, 167U, 0U, 0U, 0U, 0U, 168U, 169U, 0U, 0U,
    170U, 171U, 172U, 173U, 174U, 175U, 0U, 0U, 176U, 177U, 0U, 0U, 178U, 179U,
    0U, 0U, 0U, 0U, 180U, 181U, 182U, 183U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 184U, 185U, 0U, 0U, 0U, 0U, 0U, 0U, 186U, 187U, 0U, 0U, 188U, 189U, 0U,
    0U, 190U, 191U, 0U, 0U, 0U, 0U, 0U, 0U, 192U, 193U, 0U, 0U, 194U, 195U, 196U,
    197U, 198U, 199U, 200U, 201U, 202U, 203U, 204U, 205U, 206U, 207U, 0U, 0U, 0U,
    0U, 0U, 0U, 208U, 209U, 0U, 0U, 210U, 211U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    0U, 0U, 212U, 213U, 0U, 0U, 214U, 215U, 216U, 217U, 218U, 219U, 0U, 0U, 0U,
    0U, 220U, 221U, 0U, 0U, 0U, 0U, 222U, 223U, 224U, 225U, 0U, 0U, 0U, 0U, 0U,
    0U, 226U, 227U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U };

  static const real_T d[227] = { 0.079455266955266945, 0.53629776021080366,
    0.045715778474399164, 0.38924963924963929, 0.36730911786508524,
    0.676923076923077, 0.27115384615384619, 0.0035460992907801418,
    0.060790835181079084, 0.48038461538461541, 0.0066666666666666671,
    0.26365441906653425, 0.54740957966764414, 0.025657894736842105,
    0.04334719334719335, 0.30724070450097846, 0.68072955047899775,
    0.18708333333333332, 0.87655279503105588, 0.033615819209039548,
    0.3149535510937968, 0.0, 0.21527777777777779, 0.49108260325406761,
    0.13871910709540261, 0.0, 0.0, 0.0, 0.020300751879699246, 0.0, 0.0,
    0.13245096172157683, 0.11283565035231384, 0.99315068493150682,
    0.13580246913580246, 0.01626123744050767, 0.38855773726040205,
    0.18568241903502974, 0.73947010869565211, 0.025994318181818181,
    0.6692307692307693, 0.0, 0.72745901639344268, 0.0, 0.0, 0.222970173985087,
    0.0, 0.0, 0.02, 0.0, 0.0, 0.46992257296009532, 0.029144816458249293, 0.0,
    0.0, 0.8609375, 0.021845694799658994, 0.0, 0.0, 0.31608969315499608,
    0.61437177280550781, 0.68878865979381443, 0.0, 0.0, 0.32507976140934941, 0.0,
    0.0, 0.094281972937581837, 0.0, 0.045464982778415611, 0.0, 0.0, 0.0,
    0.77096018735363, 0.26930256236212458, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.44826132771338251, 0.57880149812734083, 0.011634199134199134,
    0.01693173212160554, 0.0, 0.49268104776579358, 0.0, 0.03919904972000679,
    0.055566274358479648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007462686567164179,
    0.14491488432998689, 0.7945314998401023, 0.0083333333333333332, 0.0,
    0.006024096385542169, 0.0, 0.0, 0.74106858054226477, 0.15239477503628446,
    0.0, 0.088415803605677024, 0.0, 0.0, 0.0, 0.0, 0.077272727272727271,
    0.12011858515640972, 0.29363399544122432, 0.074308300395256918, 0.0, 0.0,
    0.0, 0.014202256244963738, 0.44816017316017315, 0.074362041467304629, 0.0,
    0.0, 0.0, 0.0, 0.65030120481927711, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0053191489361702126, 0.0, 0.0, 0.0, 0.0, 0.0, 0.87075393537696766,
    0.034286833855799372, 0.0, 0.0, 0.0, 0.17560837577815508, 0.03125, 0.0, 0.0,
    0.25888888888888889, 0.0, 0.47127329192546585, 0.26235465116279066,
    0.595814307458143, 0.34462444771723122, 0.0, 0.0, 0.014395194697597348, 0.0,
    0.32057562767911818, 0.006024096385542169, 0.020408163265306121, 0.0,
    0.93680445151033387, 0.0, 0.085858585858585856, 0.0, 0.0,
    0.025062656641604009, 0.89333333333333331, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.41981921660528959, 0.0, 0.0, 0.0, 0.10784313725490197, 0.0,
    0.39358974358974358, 0.0, 0.78035379254891457, 0.0, 0.0, 0.0,
    0.035609551738583996, 0.0, 0.99206349206349209, 0.026097560975609758,
    0.022388059701492536, 0.1835016835016835, 0.65535827520608758,
    0.024448419797257006, 0.006024096385542169, 0.0, 0.0, 0.0, 0.99, 0.0,
    0.37331415213091168, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17132541346377822, 0.0,
    0.53178023161863719, 0.22376543209876543, 0.661527514231499, 0.0, 0.0,
    0.032051282051282048, 0.0, 0.0, 0.2784237726098191, 0.10752547687483668, 0.0,
    0.0, 0.0, 0.28855569155446759, 0.0, 0.0, 0.0, 0.0 };

  static const int8_T e[227] = { 69, 68, 64, 67, 62, 64, 61, 65, 66, 58, 38, 56,
    46, 44, 15, 60, 57, 66, 39, 50, 48, 0, 31, 54, 34, 0, 0, 0, 3, 0, 0, 32, 43,
    55, 3, 63, 52, 15, 17, 44, 33, 0, 37, 0, 0, 47, 0, 0, 19, 0, 0, 3, 22, 0, 0,
    53, 49, 0, 0, 41, 59, 24, 0, 0, 15, 0, 0, 42, 0, 22, 0, 0, 0, 45, 34, 0, 0,
    0, 0, 0, 6, 51, 40, 22, 0, 3, 0, 26, 34, 0, 0, 0, 0, 0, 0, 3, 21, 35, 8, 0,
    12, 0, 0, 36, 12, 0, 28, 0, 0, 0, 0, 12, 12, 3, 3, 0, 0, 0, 18, 23, 17, 0, 0,
    0, 0, 35, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 11, 20, 0, 0, 0, 30, 6,
    0, 0, 7, 0, 14, 20, 29, 27, 0, 0, 7, 0, 10, 6, 17, 0, 25, 0, 12, 0, 0, 5, 10,
    0, 0, 0, 0, 0, 16, 0, 0, 0, 5, 0, 10, 0, 16, 0, 0, 0, 10, 0, 13, 16, 10, 10,
    9, 13, 8, 0, 0, 0, 1, 0, 9, 0, 0, 0, 0, 0, 1, 0, 4, 9, 1, 0, 0, 4, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 0, 0, 0 };

  static const boolean_T f[227] = { false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, true, false, false, false, true, true, true,
    false, true, true, false, false, false, false, false, false, false, false,
    false, false, true, false, true, true, false, true, true, false, true, true,
    false, false, true, true, false, false, true, true, false, false, false,
    true, true, false, true, true, false, true, false, true, true, true, false,
    false, true, true, true, true, true, false, false, false, false, true, false,
    true, false, false, true, true, true, true, true, true, false, false, false,
    false, true, false, true, true, false, false, true, false, true, true, true,
    true, false, false, false, false, true, true, true, false, false, false,
    true, true, true, true, false, true, true, true, true, true, true, true,
    true, false, true, true, true, true, true, false, false, true, true, true,
    false, false, true, true, false, true, false, false, false, false, true,
    true, false, true, false, false, false, true, false, true, false, true, true,
    false, false, true, true, true, true, true, false, true, true, true, false,
    true, false, true, false, true, true, true, false, true, false, false, false,
    false, false, false, false, true, true, true, false, true, false, true, true,
    true, true, true, false, true, false, false, false, true, true, false, true,
    true, false, false, true, true, true, false, true, true, true, true };

  static const int8_T h[9] = { 0, 1, 1, 1, 0, 1, 1, 1, 0 };

  static const real_T i[681] = { 0.28789281787143539, 0.36051230027900344,
    0.13027522935778282, 0.19149561842027793, 0.79233658903082094,
    0.070237231948609286, 0.59502806736166458, 0.14221628328718314,
    0.27060986821942151, 0.86391412056153061, 0.06999999999999966,
    0.06809184481392716, 0.09359605911330128, 0.00199999999999999,
    0.99196787148594368, 0.62535612535612073, 0.067452882177882881,
    0.4070711896798837, 0.026495726495727161, 0.94432314410480844,
    0.0733452593917708, 0.0, 0.6461538461538463, 0.03870967741935033,
    0.86075949367088633, 0.0, 1.0, 0.0, 0.01428571428571427, 0.0, 1.0,
    0.956427015250544, 0.0, 0.038636363636359347, 0.99633699633699635,
    0.15446265938069603, 0.88827203331019866, 0.0052562417871223404,
    0.87719298245614052, 0.19932432432432443, 0.9867307692307703, 0.0,
    0.28275862068965557, 1.0, 0.0, 0.018680397557176466, 1.0, 1.0, 0.0, 1.0, 0.0,
    0.035714285714285691, 0.985393258426966, 0.0, 0.0, 0.0057204147300672584,
    0.71393643031784937, 0.0, 1.0, 0.0015360983102918574, 0.20200573065903157,
    0.98918083462132889, 0.0, 0.0, 0.54545454545454541, 1.0, 0.0, 0.4609375, 0.0,
    0.992072699149266, 0.0, 0.0, 1.0, 0.0040297960678954579, 0.75925925925925875,
    0.0, 0.0, 0.0, 1.0, 0.0, 0.99772468714448226, 0.0035789213871401334,
    0.0659722222222221, 0.0, 1.0, 0.0016666666666666653, 0.0,
    0.0054777845404749108, 0.91796008869179635, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.99437984496124088, 0.0040685488842307806, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.003136762860727344, 0.030075187969924869, 0.0, 0.27536231884057949, 0.0,
    0.0, 0.0, 0.5, 0.17391304347826084, 0.0030864197530865167, 0.027027027027027,
    0.99758454106280192, 0.0, 0.0, 0.0, 0.99631067961165087,
    0.003847108463638135, 0.037735849056603737, 0.0, 0.0, 0.0, 0.0,
    0.0031517902168427783, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.79999999999999993,
    0.5714285714285714, 0.00061996280223188568, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.99786075457020629, 0.0023614218245087736, 1.0, 0.0, 0.22222222222222221,
    0.0029139744077026669, 0.051282051282051232, 0.49999999999999994, 0.0,
    0.99863760217983655, 0.0, 0.0015016894005754568, 0.1272727272727272,
    0.00063726739739986937, 0.38297872340425521, 0.0, 1.0, 0.999026479750779,
    0.0, 0.0015035709810798737, 0.0, 0.26923076923076916, 0.0,
    0.000638895987733117, 0.0, 0.81818181818181823, 0.0, 0.0,
    0.99941566030385676, 0.0012532898859504609, 1.0, 0.0, 0.0, 1.0, 0.0,
    0.00064012290359741035, 0.0, 1.0, 0.0, 0.98192771084337349, 1.0,
    0.0012536041118213273, 0.0, 0.00064045087741762161, 0.0, 0.0, 1.0,
    0.0011285266457678814, 0.5, 0.00025916807049368266, 0.033333333333333312,
    0.0, 0.0011525163273144914, 0.00026007802340698947, 0.0, 0.15789473684210528,
    0.0, 0.0, 0.0, 0.00064061499039069422, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.00064102564102556, 0.0, 0.0, 0.0, 0.00051301782736943625,
    0.33333333333333331, 0.0, 0.0, 0.0, 0.0, 0.00025843132187617881,
    0.03448275862068962, 0.0, 0.0, 0.0, 0.025974025974025955,
    0.39999999999999997, 0.0, 0.0, 1.0, 0.34180218762659303, 0.44133062811730467,
    0.12577981651374676, 0.56989942951243378, 0.1128474830954074,
    0.097482647881475662, 0.34482758620689968, 0.765963539181091,
    0.25513331290222951, 0.035672997522705627, 0.891666666666667,
    0.023187422237300807, 0.90640394088669862, 0.86000000000000054, 0.0,
    0.0690883190883196, 0.87380138873582336, 0.39775441949354851, 0.0,
    0.020378457059677904, 0.18604651162790603, 1.0, 0.0, 0.019941348973604629,
    0.11075949367088582, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.031590413943355614,
    0.13991769547325053, 0.90079545454546539, 0.00366300366300365,
    0.60655737704917267, 0.0, 0.0, 0.0, 0.23310810810810823,
    0.0082692307692300085, 0.0, 0.71724137931034437, 0.0, 0.0,
    0.020356843491794882, 0.0, 0.0, 0.79545454545454564, 0.0, 0.0,
    0.9642857142857143, 0.0022471910112359895, 0.0, 1.0, 0.932070075080451,
    0.25916870415647825, 1.0, 0.0, 0.079877112135176676, 0.77029608404965921,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5390625, 0.0, 0.0029002320185612142, 1.0,
    1.0, 0.0, 0.020271095371837741, 0.024691358024691384, 1.0, 0.0, 1.0, 0.0,
    0.0, 0.0022753128555176691, 0.9590275206713611, 0.17361111111111097,
    0.90598290598290587, 0.0, 0.0016666666666666653, 1.0, 0.98113207547169745,
    0.0022172949002217182, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.91666666666666663,
    0.00077519379844953986, 0.011219331771666684, 0.96153846153846156, 0.0,
    0.10256410256410248, 0.0, 1.0, 0.97503136762861031, 0.0, 0.0,
    0.7246376811594204, 0.0, 1.0, 0.0, 0.5, 0.043478260869565209,
    0.99444444444444424, 0.0, 0.0024154589371980545, 1.0, 0.0, 0.0,
    0.00077669902912614079, 0.0058327128319674969, 0.83018867924528317, 1.0, 0.0,
    1.0, 0.0, 0.97970247100353258, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.19999999999999998, 0.0, 0.99876007439553627, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.000777907429015874, 0.0058414118816796029, 0.0, 1.0, 0.0,
    0.98454326618522947, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0035039419346760679,
    0.34545454545454535, 0.98993117512108209, 0.085106382978723374, 0.0, 0.0,
    0.0, 0.0, 0.0025059516351331234, 0.79999999999999993, 0.73076923076923073,
    0.0, 0.99246102734474928, 0.0, 0.1818181818181818, 0.0, 0.0, 0.0,
    0.0025065797719009222, 0.0, 1.0, 0.0, 0.0, 1.0, 0.9943669184483428, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0022564874012783892, 1.0, 0.994876392980659, 0.0, 0.0, 0.0,
    0.0021316614420059982, 0.5, 0.99727873525981636, 0.78888888888888908,
    0.072289156626506076, 0.00064028684850805082, 0.998439531859558,
    0.66666666666666663, 0.0, 1.0, 1.0, 0.0, 0.00064061499039069422, 0.0,
    0.99869927159209171, 0.0, 0.0, 1.0, 0.0, 0.0, 0.00038461538461533604,
    0.39999999999999997, 0.99973294164775006, 0.95979899497487431,
    0.00038476337052707716, 0.0, 1.0, 0.84615384615384615, 0.0, 1.0, 0.0,
    0.051724137931034427, 0.0, 1.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.37030499450197163, 0.19815707160369198, 0.74394495412847028,
    0.23860495206728824, 0.0948159278737718, 0.83228012016991515,
    0.060144346431435687, 0.091820177531726, 0.47425681887834886,
    0.10041288191576375, 0.038333333333333129, 0.90872073294877209, 0.0,
    0.13799999999999946, 0.008032128514056297, 0.30555555555555974,
    0.058745729086293816, 0.19517439082656779, 0.97350427350427282,
    0.035298398835513611, 0.74060822898032308, 0.0, 0.3538461538461537,
    0.94134897360704506, 0.02848101265822776, 0.0, 0.0, 0.0, 0.98571428571428577,
    1.0, 0.0, 0.011982570806100398, 0.8600823045267495, 0.0605681818181752, 0.0,
    0.23897996357013127, 0.11172796668980135, 0.99474375821287775,
    0.12280701754385953, 0.56756756756756721, 0.0049999999999995378, 1.0, 0.0,
    0.0, 1.0, 0.96096275895102856, 0.0, 0.0, 0.20454545454545442, 0.0, 1.0, 0.0,
    0.012359550561797942, 1.0, 0.0, 0.062209510189481752, 0.026894865525672305,
    0.0, 0.0, 0.91858678955453144, 0.027698185291309175, 0.010819165378671072,
    1.0, 1.0, 0.45454545454545453, 0.0, 1.0, 0.0, 1.0, 0.0050270688321727743,
    0.0, 0.0, 0.0, 0.97569910856026676, 0.21604938271604981, 0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.037393557941498762, 0.76041666666666685, 0.094017094017094127,
    0.0, 0.99666666666666659, 0.0, 0.013390139987827563, 0.079822616407981925,
    0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.083333333333333343, 0.0048449612403096273,
    0.98471211934410252, 0.038461538461538443, 0.0, 0.89743589743589758, 0.0,
    0.0, 0.021831869510662311, 0.96992481203007508, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    0.78260869565217384, 0.0024691358024692134, 0.97297297297297292, 0.0, 0.0,
    1.0, 1.0, 0.0029126213592230278, 0.99032017870439437, 0.13207547169811309,
    0.0, 1.0, 0.0, 1.0, 0.017145738779624671, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.4285714285714286, 0.00061996280223188568, 1.0, 0.0, 0.0, 0.0, 1.0,
    0.0013613380007777794, 0.99179716629381165, 0.0, 0.0, 0.77777777777777768,
    0.012542759407067984, 0.94871794871794868, 0.5, 0.0, 0.0013623978201633595,
    0.0, 0.99499436866474844, 0.52727272727272734, 0.0094315574815180738,
    0.53191489361702138, 1.0, 0.0, 0.00097352024922109229, 1.0,
    0.995990477383787, 0.2, 0.0, 1.0, 0.0069000766675176711, 1.0, 0.0, 1.0, 1.0,
    0.00058433969614330318, 0.99624013034214864, 0.0, 0.0, 1.0, 0.0, 0.0,
    0.0049929586480598051, 1.0, 0.0, 0.0, 0.018072289156626512, 0.0,
    0.99648990848690033, 0.0, 0.0044831561419233553, 1.0, 1.0, 0.0,
    0.99673981191222616, 0.0, 0.0024620966696899857, 0.17777777777777765,
    0.927710843373494, 0.99820719682417747, 0.0013003901170349475,
    0.33333333333333331, 0.8421052631578948, 0.0, 0.0, 1.0, 0.9987187700192186,
    0.0, 0.0013007284079082657, 0.0, 1.0, 0.0, 1.0, 0.0, 0.99897435897435916,
    0.60000000000000009, 0.00026705835224993352, 0.04020100502512558,
    0.99910221880210348, 0.66666666666666674, 0.0, 0.15384615384615385, 1.0, 0.0,
    0.9997415686781238, 0.91379310344827591, 1.0, 0.0, 1.0, 0.974025974025974,
    0.0, 1.0, 1.0, 0.0 };

  boolean_T exitg1;

  /* Outport: '<Root>/pos' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.pos = -nczallcode_DW.VARN;
  for (k = 0; k < 160; k++) {
    /* MATLAB Function: '<Root>/MATLAB Function' incorporates:
     *  Inport: '<Root>/frameData'
     */
    frameData[k] = roundf(nczallcode_U.frameData[k] / 6.0F);

    /* Outport: '<Root>/frame_img' incorporates:
     *  MATLAB Function: '<Root>/MATLAB Function'
     */
    nczallcode_Y.frame_img[k] = -nczallcode_DW.VARN;
  }

  /* MATLAB Function: '<Root>/MATLAB Function' incorporates:
   *  Inport: '<Root>/frameData'
   */
  nczallcode_DW.SEND_DATA_N++;
  if (nczallcode_DW.SEND_DATA_N >= 4.0F) {
    if (!nczallcode_DW.treeModel_not_empty) {
      for (k = 0; k < 227; k++) {
        nczallcode_DW.treeModel.CutPredictorIndex[k] = b[k];
      }

      for (k = 0; k < 454; k++) {
        nczallcode_DW.treeModel.Children[k] = c[k];
      }

      for (k = 0; k < 227; k++) {
        nczallcode_DW.treeModel.CutPoint[k] = d[k];
        nczallcode_DW.treeModel.PruneList[k] = e[k];
        nczallcode_DW.treeModel.NanCutPoints[k] = f[k];
        nczallcode_DW.treeModel.InfCutPoints[k] = false;
      }

      nczallcode_DW.treeModel.ClassNamesLength[0] = 1;
      nczallcode_DW.treeModel.Prior[0] = 0.28789281787140458;
      nczallcode_DW.treeModel.ClassNamesLength[1] = 1;
      nczallcode_DW.treeModel.Prior[1] = 0.34180218762659875;
      nczallcode_DW.treeModel.ClassNamesLength[2] = 1;
      nczallcode_DW.treeModel.Prior[2] = 0.37030499450199666;
      for (k = 0; k < 9; k++) {
        nczallcode_DW.treeModel.Cost[k] = h[k];
      }

      nczallcode_DW.treeModel.CharClassNamesLength[0] = 1;
      nczallcode_DW.treeModel.CharClassNamesLength[1] = 1;
      nczallcode_DW.treeModel.CharClassNamesLength[2] = 1;
      memcpy(&nczallcode_DW.treeModel.ClassProbability[0], &i[0], 681U * sizeof
             (real_T));
      nczallcode_DW.treeModel_not_empty = true;
    }

    tmp_0 = !rtIsNaNF(nczallcode_U.frameData[0]);
    if (tmp_0) {
      k = 1;
    } else {
      k = 0;
      b_k = 2;
      exitg1 = false;
      while ((!exitg1) && (b_k < 161)) {
        if (!rtIsNaNF(nczallcode_U.frameData[b_k - 1])) {
          k = b_k;
          exitg1 = true;
        } else {
          b_k++;
        }
      }
    }

    if (k == 0) {
      X_min = nczallcode_U.frameData[0];
    } else {
      X_min = nczallcode_U.frameData[k - 1];
      for (b_k = k + 1; b_k < 161; b_k++) {
        frameData_0 = nczallcode_U.frameData[b_k - 1];
        if (X_min > frameData_0) {
          X_min = frameData_0;
        }
      }
    }

    if (tmp_0) {
      k = 1;
    } else {
      k = 0;
      b_k = 2;
      exitg1 = false;
      while ((!exitg1) && (b_k < 161)) {
        if (!rtIsNaNF(nczallcode_U.frameData[b_k - 1])) {
          k = b_k;
          exitg1 = true;
        } else {
          b_k++;
        }
      }
    }

    if (k == 0) {
      X_max = nczallcode_U.frameData[0];
    } else {
      X_max = nczallcode_U.frameData[k - 1];
      for (b_k = k + 1; b_k < 161; b_k++) {
        frameData_0 = nczallcode_U.frameData[b_k - 1];
        if (X_max < frameData_0) {
          X_max = frameData_0;
        }
      }
    }

    frameData_0 = (X_max - X_min) + 2.22044605E-16F;
    for (k = 0; k < 160; k++) {
      tmp[k] = (nczallcode_U.frameData[k] - X_min) / frameData_0;
    }

    a_codes = ncza_ClassificationTree_predict
      (nczallcode_DW.treeModel.CutPredictorIndex,
       nczallcode_DW.treeModel.Children, nczallcode_DW.treeModel.CutPoint,
       nczallcode_DW.treeModel.PruneList, nczallcode_DW.treeModel.NanCutPoints,
       nczallcode_DW.treeModel.Prior, nczallcode_DW.treeModel.Cost,
       nczallcode_DW.treeModel.ClassProbability, tmp);

    /* Outport: '<Root>/pos' incorporates:
     *  Inport: '<Root>/frameData'
     */
    nczallcode_Y.pos = a_codes;
    if (a_codes == 0) {
      /* Outport: '<Root>/pos' */
      nczallcode_Y.pos = (rtNaNF);
    }

    nczallcode_DW.BREATH_MIN += nczallcode_DW.BREATH_RATE;
    nczallcode_DW.SEND_DATA_N = nczallcode_DW.VARN - 1.0F;
  }

  if (nczallcode_DW.MIN_SEND_N >= 720.0F) {
    /* Outport: '<Root>/frame_img' */
    memcpy(&nczallcode_Y.frame_img[0], &frameData[0], 160U * sizeof(real32_T));
  }

  /* Outport: '<Root>/bodyMovementData' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  for (k = 0; k < 24; k++) {
    nczallcode_Y.bodyMovementData[k] = -nczallcode_DW.VARN;
  }

  /* End of Outport: '<Root>/bodyMovementData' */

  /* Outport: '<Root>/rate' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.rate = -nczallcode_DW.VARN;

  /* Outport: '<Root>/strokerisk' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.strokerisk = -nczallcode_DW.VARN;

  /* Outport: '<Root>/stateInBbed' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.stateInBbed = -nczallcode_DW.VARN;

  /* Outport: '<Root>/inBedtime' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.inBedtime = -nczallcode_DW.VARN;

  /* Outport: '<Root>/rateMin' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.rateMin = -nczallcode_DW.VARN;

  /* Outport: '<Root>/strokeriskMin' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.strokeriskMin = -nczallcode_DW.VARN;

  /* Outport: '<Root>/statesleep' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  for (k = 0; k < 48; k++) {
    nczallcode_Y.statesleep[k] = -nczallcode_DW.VARN;
  }

  /* End of Outport: '<Root>/statesleep' */

  /* Outport: '<Root>/timt' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.timt = -nczallcode_DW.VARN;

  /* Outport: '<Root>/meansn' incorporates:
   *  MATLAB Function: '<Root>/MATLAB Function'
   */
  nczallcode_Y.meansn = -nczallcode_DW.VARN;
}

/* Model initialize function */
void nczallcode_initialize(void)
{
  /* Start for DataStoreMemory: '<Root>/Data Store Memory19' */
  nczallcode_DW.VARN = 1.0F;
}

/* Model terminate function */
void nczallcode_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
