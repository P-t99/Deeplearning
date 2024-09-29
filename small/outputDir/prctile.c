/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: prctile.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

/* Include Files */
#include "prctile.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Declarations */
static double rt_roundd_snf(double u);

/* Function Definitions */
/*
 * Arguments    : double u
 * Return Type  : double
 */
static double rt_roundd_snf(double u)
{
  double y;
  if (fabs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = ceil(u - 0.5);
    }
  } else {
    y = u;
  }
  return y;
}

/*
 * Arguments    : const float x[160]
 *                double p
 * Return Type  : float
 */
float percentile_vector(const float x[160], double p)
{
  double b_i;
  double r;
  float f;
  float pct;
  int idx[160];
  int iwork[160];
  int b_p;
  int c_i;
  int i;
  int i2;
  int j;
  int kEnd;
  int nj;
  int pEnd;
  int q;
  int qEnd;
  for (nj = 0; nj <= 158; nj += 2) {
    f = x[nj + 1];
    if ((x[nj] <= f) || rtIsNaNF(f)) {
      idx[nj] = nj + 1;
      idx[nj + 1] = nj + 2;
    } else {
      idx[nj] = nj + 2;
      idx[nj + 1] = nj + 1;
    }
  }
  i = 2;
  while (i < 160) {
    i2 = i << 1;
    j = 1;
    for (pEnd = i + 1; pEnd < 161; pEnd = qEnd + i) {
      b_p = j;
      q = pEnd - 1;
      qEnd = j + i2;
      if (qEnd > 161) {
        qEnd = 161;
      }
      nj = 0;
      kEnd = qEnd - j;
      while (nj + 1 <= kEnd) {
        f = x[idx[q] - 1];
        c_i = idx[b_p - 1];
        if ((x[c_i - 1] <= f) || rtIsNaNF(f)) {
          iwork[nj] = c_i;
          b_p++;
          if (b_p == pEnd) {
            while (q + 1 < qEnd) {
              nj++;
              iwork[nj] = idx[q];
              q++;
            }
          }
        } else {
          iwork[nj] = idx[q];
          q++;
          if (q + 1 == qEnd) {
            while (b_p < pEnd) {
              nj++;
              iwork[nj] = idx[b_p - 1];
              b_p++;
            }
          }
        }
        nj++;
      }
      for (nj = 0; nj < kEnd; nj++) {
        idx[(j + nj) - 1] = iwork[nj];
      }
      j = qEnd;
    }
    i = i2;
  }
  nj = 160;
  while ((nj > 0) && rtIsNaNF(x[idx[nj - 1] - 1])) {
    nj--;
  }
  if (nj < 1) {
    pct = rtNaNF;
  } else if (nj == 1) {
    pct = x[idx[0] - 1];
  } else {
    r = p / 100.0 * (double)nj;
    b_i = rt_roundd_snf(r);
    if (b_i < 1.0) {
      pct = x[idx[0] - 1];
    } else if (b_i >= nj) {
      pct = x[idx[nj - 1] - 1];
    } else {
      r -= b_i;
      pct = (float)(0.5 - r) * x[idx[(int)b_i - 1] - 1] +
            (float)(r + 0.5) * x[idx[(int)(b_i + 1.0) - 1] - 1];
    }
  }
  return pct;
}

/*
 * File trailer for prctile.c
 *
 * [EOF]
 */
