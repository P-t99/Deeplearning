/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: predictTree.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

/* Include Files */
#include "predictTree.h"
#include "ClassificationTree.h"
#include "prctile.h"
#include "predictTree_data.h"
#include "predictTree_initialize.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Type Definitions */
#ifndef typedef_ClassificationTree
#define typedef_ClassificationTree
typedef struct {
  float CutPredictorIndex[65];
  float Children[130];
  float CutPoint[65];
  float PruneList[65];
  boolean_T NanCutPoints[65];
  float Cost[9];
  float ClassProbability[195];
} ClassificationTree;
#endif /* typedef_ClassificationTree */

/* Variable Definitions */
static boolean_T treeModel_not_empty;

/* Function Declarations */
static float rt_roundf_snf(float u);

/* Function Definitions */
/*
 * Arguments    : float u
 * Return Type  : float
 */
static float rt_roundf_snf(float u)
{
  float y;
  if ((float)fabs(u) < 8.388608E+6F) {
    if (u >= 0.5F) {
      y = (float)floor(u + 0.5F);
    } else if (u > -0.5F) {
      y = u * 0.0F;
    } else {
      y = (float)ceil(u - 0.5F);
    }
  } else {
    y = u;
  }
  return y;
}

/*
 * 加载预训练的决策树模型
 *
 * Arguments    : const double XTest[160]
 * Return Type  : float
 */
float predictTree(const double XTest[160])
{
  static ClassificationTree treeModel;
  static const float fv1[195] = {0.287053436F,
                                 0.35654816F,
                                 0.145565212F,
                                 0.18868582F,
                                 0.764930129F,
                                 0.0633662F,
                                 0.626351416F,
                                 0.0965254754F,
                                 0.291757137F,
                                 0.900387943F,
                                 0.0202117451F,
                                 0.0644422695F,
                                 0.0505320653F,
                                 0.00335008022F,
                                 0.975586653F,
                                 0.576641262F,
                                 0.0169395171F,
                                 0.385826588F,
                                 0.0576050058F,
                                 0.955755234F,
                                 0.00600601174F,
                                 0.0515970178F,
                                 0.0F,
                                 0.0262329094F,
                                 0.578099906F,
                                 0.0F,
                                 1.0F,
                                 0.0128205167F,
                                 0.0F,
                                 0.00477326429F,
                                 0.871006906F,
                                 0.0173071455F,
                                 0.0F,
                                 0.188939944F,
                                 0.874685526F,
                                 0.00334923342F,
                                 0.916666687F,
                                 0.997279525F,
                                 0.0386266448F,
                                 0.00623451F,
                                 1.0F,
                                 0.00769231096F,
                                 0.988919675F,
                                 0.0212765858F,
                                 0.981944382F,
                                 0.0121470522F,
                                 0.330508351F,
                                 0.00817440357F,
                                 0.230286136F,
                                 0.0552764572F,
                                 0.992080688F,
                                 0.169811338F,
                                 0.0F,
                                 0.00627980288F,
                                 0.0F,
                                 0.0F,
                                 0.0115607036F,
                                 1.0F,
                                 0.0F,
                                 0.00881059282F,
                                 0.0F,
                                 0.0985585898F,
                                 0.944000185F,
                                 0.00629841629F,
                                 0.0F,
                                 0.340384305F,
                                 0.444358915F,
                                 0.128697708F,
                                 0.580372691F,
                                 0.113461621F,
                                 0.105335057F,
                                 0.265344918F,
                                 0.82171011F,
                                 0.310459971F,
                                 0.0665248111F,
                                 0.3715114F,
                                 0.0345623679F,
                                 0.949467957F,
                                 0.738693833F,
                                 0.0F,
                                 0.0851583257F,
                                 0.943803966F,
                                 0.433016539F,
                                 0.0054004658F,
                                 0.00910923537F,
                                 0.993994057F,
                                 0.948403F,
                                 0.0F,
                                 0.0170095097F,
                                 0.270531237F,
                                 1.0F,
                                 0.0F,
                                 0.0F,
                                 1.0F,
                                 0.00477326429F,
                                 0.126536041F,
                                 0.964286864F,
                                 0.0F,
                                 0.606143F,
                                 0.00314861163F,
                                 0.000478461938F,
                                 0.0833332837F,
                                 0.0009715989F,
                                 0.188841358F,
                                 0.0173588358F,
                                 0.0F,
                                 0.646153569F,
                                 0.0F,
                                 0.978723407F,
                                 0.0152778253F,
                                 0.980173767F,
                                 0.0F,
                                 0.0722072423F,
                                 0.728268385F,
                                 0.0F,
                                 0.00359970843F,
                                 0.830188692F,
                                 0.0F,
                                 0.010220075F,
                                 1.0F,
                                 0.0F,
                                 0.971098244F,
                                 0.0F,
                                 0.0F,
                                 0.0F,
                                 1.0F,
                                 0.861944F,
                                 0.00399999227F,
                                 0.00728640379F,
                                 1.0F,
                                 0.372562259F,
                                 0.19909288F,
                                 0.725737095F,
                                 0.230941489F,
                                 0.12160819F,
                                 0.831298709F,
                                 0.108303741F,
                                 0.0817643553F,
                                 0.397782892F,
                                 0.0330872759F,
                                 0.608276844F,
                                 0.900995314F,
                                 0.0F,
                                 0.257956088F,
                                 0.0244133659F,
                                 0.33820042F,
                                 0.0392565429F,
                                 0.181156904F,
                                 0.936994553F,
                                 0.0351355672F,
                                 0.0F,
                                 0.0F,
                                 1.0F,
                                 0.956757605F,
                                 0.151368871F,
                                 0.0F,
                                 0.0F,
                                 0.987179458F,
                                 0.0F,
                                 0.990453482F,
                                 0.00245700916F,
                                 0.018406013F,
                                 1.0F,
                                 0.204917118F,
                                 0.122165933F,
                                 0.996172309F,
                                 0.0F,
                                 0.00174887793F,
                                 0.772532F,
                                 0.976406634F,
                                 0.0F,
                                 0.346154153F,
                                 0.011080334F,
                                 0.0F,
                                 0.00277778646F,
                                 0.0076791686F,
                                 0.669491708F,
                                 0.919618368F,
                                 0.0414454676F,
                                 0.944723547F,
                                 0.00431965F,
                                 0.0F,
                                 1.0F,
                                 0.983500123F,
                                 0.0F,
                                 1.0F,
                                 0.017341055F,
                                 0.0F,
                                 1.0F,
                                 0.99118942F,
                                 0.0F,
                                 0.039497409F,
                                 0.0519998856F,
                                 0.986415207F,
                                 0.0F};
  static const float fv[65] = {
      22.5F,  176.5F, 11.5F,  92.5F,  226.5F, 197.5F, 84.5F,  1.5F,   26.5F,
      167.0F, 26.0F,  181.5F, 107.5F, 17.5F,  0.0F,   1.5F,   201.5F, 62.5F,
      240.5F, 128.5F, 0.0F,   0.0F,   0.0F,   163.5F, 191.5F, 0.0F,   0.0F,
      0.0F,   0.0F,   0.0F,   7.5F,   214.5F, 0.0F,   16.5F,  17.5F,  0.0F,
      0.0F,   0.0F,   20.0F,  113.5F, 0.0F,   140.0F, 0.0F,   0.0F,   0.0F,
      0.0F,   207.0F, 92.5F,  6.5F,   0.0F,   0.0F,   0.0F,   0.0F,   247.5F,
      0.0F,   0.0F,   0.0F,   0.0F,   0.0F,   0.0F,   0.0F,   0.0F,   0.0F,
      0.0F,   0.0F};
  static const signed char iv[130] = {
      2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
      19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 0,  0,  30, 31, 32, 33,
      34, 35, 36, 37, 38, 39, 0,  0,  0,  0,  0,  0,  40, 41, 42, 43, 0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  44, 45, 46, 47, 0,  0,  48, 49,
      50, 51, 0,  0,  0,  0,  0,  0,  52, 53, 54, 55, 0,  0,  56, 57, 0,
      0,  0,  0,  0,  0,  0,  0,  58, 59, 60, 61, 62, 63, 0,  0,  0,  0,
      0,  0,  0,  0,  64, 65, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};
  static const signed char iv1[65] = {
      30, 29, 27, 28, 24, 26, 21, 23, 28, 18, 19, 17, 2, 11, 0,  20, 12,
      28, 10, 15, 0,  0,  0,  14, 13, 0,  0,  0,  0,  0, 9,  7,  0,  25,
      16, 0,  0,  0,  4,  6,  0,  8,  0,  0,  0,  0,  3, 5,  22, 0,  0,
      0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0};
  static const unsigned char uv[65] = {
      103U, 66U, 147U, 77U, 77U, 17U, 65U,  106U, 115U, 117U, 95U,  26U, 21U,
      29U,  0U,  49U,  42U, 94U, 46U, 105U, 0U,   0U,   0U,   131U, 65U, 0U,
      0U,   0U,  0U,   0U,  23U, 23U, 0U,   75U,  97U,  0U,   0U,   0U,  5U,
      69U,  0U,  84U,  0U,  0U,  0U,  0U,   24U,  108U, 150U, 0U,   0U,  0U,
      0U,   82U, 0U,   0U,  0U,  0U,  0U,   0U,   0U,   0U,   0U,   0U,  0U};
  static const signed char iv2[9] = {0, 1, 1, 1, 0, 1, 1, 1, 0};
  static const boolean_T bv[65] = {
      false, false, false, false, false, false, false, false, false, false,
      false, false, false, false, true,  false, false, false, false, false,
      true,  true,  true,  false, false, true,  true,  true,  true,  true,
      false, false, true,  false, false, true,  true,  true,  false, false,
      true,  false, true,  true,  true,  true,  false, false, false, true,
      true,  true,  true,  false, true,  true,  true,  true,  true,  true,
      true,  true,  true,  true,  true};
  float X[160];
  float f;
  float lower_percentile;
  float y;
  int i;
  if (!isInitialized_outputFileName) {
    predictTree_initialize();
  }
  if (!treeModel_not_empty) {
    for (i = 0; i < 65; i++) {
      treeModel.CutPredictorIndex[i] = uv[i];
    }
    for (i = 0; i < 130; i++) {
      treeModel.Children[i] = iv[i];
    }
    for (i = 0; i < 65; i++) {
      treeModel.CutPoint[i] = fv[i];
      treeModel.PruneList[i] = iv1[i];
      treeModel.NanCutPoints[i] = bv[i];
    }
    for (i = 0; i < 9; i++) {
      treeModel.Cost[i] = iv2[i];
    }
    for (i = 0; i < 195; i++) {
      treeModel.ClassProbability[i] = fv1[i];
    }
    treeModel_not_empty = true;
  }
  /*  归一化输入数据 */
  for (i = 0; i < 160; i++) {
    X[i] = (float)XTest[i];
  }
  /*  转换为单精度 */
  lower_percentile = percentile_vector(X, 2.0);
  /*  归一化到 0-255 范围并取整 */
  y = percentile_vector(X, 98.0) - lower_percentile;
  for (i = 0; i < 160; i++) {
    f = (X[i] - lower_percentile) / y * 255.0F;
    if ((f >= 255.0F) || rtIsNaNF(f)) {
      f = 255.0F;
    }
    if (f <= 0.0F) {
      f = 0.0F;
    }
    X[i] = rt_roundf_snf(f);
  }
  /*  使用归一化后的数据进行预测 */
  return ClassificationTree_predict(
      treeModel.CutPredictorIndex, treeModel.Children, treeModel.CutPoint,
      treeModel.PruneList, treeModel.NanCutPoints, treeModel.Cost,
      treeModel.ClassProbability, X);
}

/*
 * 加载预训练的决策树模型
 *
 * Arguments    : void
 * Return Type  : void
 */
void predictTree_init(void)
{
  treeModel_not_empty = false;
}

/*
 * File trailer for predictTree.c
 *
 * [EOF]
 */
