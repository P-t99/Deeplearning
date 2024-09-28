/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ClassificationTree.c
 *
 * MATLAB Coder version            : 24.1
 * C/C++ source code generated on  : 2024-09-28 13:15:57
 */

/* Include Files */
#include "ClassificationTree.h"
#include "categorical.h"
#include "predictTree_types.h"
#include "rt_nonfinite.h"
#include "strtrim.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const double obj_CutPredictorIndex[227]
 *                const double obj_Children[454]
 *                const double obj_CutPoint[227]
 *                const double obj_PruneList[227]
 *                const bool obj_NanCutPoints[227]
 *                const double obj_Prior[3]
 *                const double obj_Cost[9]
 *                const double obj_ClassProbability[681]
 *                const double Xin[160]
 *                cell_wrap_1 labels_categoryNames[3]
 * Return Type  : unsigned char
 */
unsigned char ClassificationTree_predict(
    const double obj_CutPredictorIndex[227], const double obj_Children[454],
    const double obj_CutPoint[227], const double obj_PruneList[227],
    const bool obj_NanCutPoints[227], const double obj_Prior[3],
    const double obj_Cost[9], const double obj_ClassProbability[681],
    const double Xin[160], cell_wrap_1 labels_categoryNames[3])
{
  static const char cv[11] = {'<', 'u', 'n', 'd', 'e', 'f',
                              'i', 'n', 'e', 'd', '>'};
  static const char b_names[3] = {'1', '2', '3'};
  cell_wrap_1 names[4];
  cell_wrap_1 inData;
  cell_wrap_1 labels;
  cell_wrap_1 tempnames;
  double a__4[3];
  double d;
  double ex;
  int tmp_size[2];
  int i;
  int idx;
  int iindx;
  int k;
  int m;
  char tmp_data[11];
  bool b[3];
  bool exitg1;
  bool y;
  m = 0;
  exitg1 = false;
  while (!(exitg1 || (obj_PruneList[m] <= 0.0))) {
    d = Xin[(int)obj_CutPredictorIndex[m] - 1];
    if (rtIsNaN(d) || obj_NanCutPoints[m]) {
      exitg1 = true;
    } else if (d < obj_CutPoint[m]) {
      m = (int)obj_Children[m << 1] - 1;
    } else {
      m = (int)obj_Children[(m << 1) + 1] - 1;
    }
  }
  for (i = 0; i < 3; i++) {
    a__4[i] = (obj_ClassProbability[m] * obj_Cost[3 * i] +
               obj_ClassProbability[m + 227] * obj_Cost[3 * i + 1]) +
              obj_ClassProbability[m + 454] * obj_Cost[3 * i + 2];
  }
  ex = obj_Prior[0];
  iindx = 1;
  if (obj_Prior[0] < obj_Prior[1]) {
    ex = obj_Prior[1];
    iindx = 2;
  }
  if (ex < obj_Prior[2]) {
    iindx = 3;
  }
  if (!rtIsNaN(a__4[0])) {
    idx = 1;
  } else {
    idx = 0;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k < 4)) {
      if (!rtIsNaN(a__4[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx == 0) {
    idx = 1;
  } else {
    ex = a__4[idx - 1];
    i = idx + 1;
    for (k = i; k < 4; k++) {
      d = a__4[k - 1];
      if (ex > d) {
        ex = d;
        idx = k;
      }
    }
  }
  b[0] = rtIsNaN(obj_ClassProbability[m]);
  b[1] = rtIsNaN(obj_ClassProbability[m + 227]);
  b[2] = rtIsNaN(obj_ClassProbability[m + 454]);
  y = true;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 3)) {
    if (!b[k]) {
      y = false;
      exitg1 = true;
    } else {
      k++;
    }
  }
  names[0].f1.size[0] = 1;
  names[0].f1.size[1] = 11;
  for (i = 0; i < 11; i++) {
    names[0].f1.data[i] = cv[i];
  }
  names[1].f1.size[0] = 1;
  names[1].f1.size[1] = 1;
  names[1].f1.data[0] = '1';
  names[2].f1.size[0] = 1;
  names[2].f1.size[1] = 1;
  names[2].f1.data[0] = '2';
  names[3].f1.size[0] = 1;
  names[3].f1.size[1] = 1;
  names[3].f1.data[0] = '3';
  tempnames.f1.size[0] = 1;
  tempnames.f1.size[1] = 1;
  tempnames.f1.data[0] = names[iindx].f1.data[0];
  if (!y) {
    tempnames.f1.size[0] = 1;
    tempnames.f1.size[1] = 1;
    tempnames.f1.data[0] = b_names[idx - 1];
  }
  strtrim(tempnames.f1.data, tempnames.f1.size, inData.f1.data, inData.f1.size);
  i = 0;
  strtrim(tempnames.f1.data, tempnames.f1.size, tmp_data, tmp_size);
  y = false;
  if (((signed char)tmp_size[1] == 1) && (inData.f1.data[0] == '1')) {
    y = true;
  }
  if (y) {
    i = 1;
  } else {
    y = false;
    if (((signed char)tmp_size[1] == 1) && (inData.f1.data[0] == '2')) {
      y = true;
    }
    if (y) {
      i = 2;
    } else {
      y = false;
      if (((signed char)tmp_size[1] == 1) && (inData.f1.data[0] == '3')) {
        y = true;
      }
      if (y) {
        i = 3;
      }
    }
  }
  names[0].f1.size[0] = 1;
  names[0].f1.size[1] = 11;
  for (iindx = 0; iindx < 11; iindx++) {
    names[0].f1.data[iindx] = cv[iindx];
  }
  names[1].f1.size[0] = 1;
  names[1].f1.size[1] = 1;
  names[1].f1.data[0] = '1';
  names[2].f1.size[0] = 1;
  names[2].f1.size[1] = 1;
  names[2].f1.data[0] = '2';
  names[3].f1.size[0] = 1;
  names[3].f1.size[1] = 1;
  names[3].f1.data[0] = '3';
  labels.f1.size[0] = 1;
  m = names[i].f1.size[1];
  labels.f1.size[1] = m;
  for (iindx = 0; iindx < m; iindx++) {
    labels.f1.data[iindx] = names[i].f1.data[iindx];
  }
  cell_wrap_1 b_tempnames[3];
  b_tempnames[0].f1.size[0] = 1;
  b_tempnames[0].f1.size[1] = 1;
  b_tempnames[0].f1.data[0] = '1';
  b_tempnames[1].f1.size[0] = 1;
  b_tempnames[1].f1.size[1] = 1;
  b_tempnames[1].f1.data[0] = '2';
  b_tempnames[2].f1.size[0] = 1;
  b_tempnames[2].f1.size[1] = 1;
  b_tempnames[2].f1.data[0] = '3';
  return categorical_categorical(labels, b_tempnames, labels_categoryNames);
}

/*
 * File trailer for ClassificationTree.c
 *
 * [EOF]
 */
