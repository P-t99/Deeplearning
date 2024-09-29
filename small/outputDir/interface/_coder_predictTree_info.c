/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_predictTree_info.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 29-Sep-2024 14:09:21
 */

/* Include Files */
#include "_coder_predictTree_info.h"
#include "emlrt.h"
#include "tmwtypes.h"

/* Function Declarations */
static const mxArray *c_emlrtMexFcnResolvedFunctionsI(void);

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : const mxArray *
 */
static const mxArray *c_emlrtMexFcnResolvedFunctionsI(void)
{
  const mxArray *nameCaptureInfo;
  const char_T *data[5] = {
      "789ce554cd6adb40105e352e14421293439fc0a4078348426993d09315121c3bc6c532a1"
      "780d594be358747fc4eeba386f905b5fa1c73c42c873e4966709d496"
      "2d591608991864dacc6576f866f7fb7676679051bd341042db686af79b53bf358b8b33ff"
      "0e2d5a12371279c6623a7a8f0a0bfb42fcf7cc3b826b18e969c00983",
      "68a72b98c709d7f6ad0f488212f417b801d2f728d81e83563c684c22761683a260024dd6"
      "d6009c9fad214372a0e60a693c88ea719d72df42463d9296ac4732ef"
      "adf0fd79255f78feb70cbe10efb4bba72758822f94a785bcc52e804f8148eef11bac18a1"
      "14f7c7ff8a36e590836b4b804be1023519d171bdd7297a7696d49bf4",
      "f3fc0f81ffb2fb6ce4c9f7a972bc91275f68ebe21ba59cb7ec7ffb98c2574ce035f65d7d"
      "3df59dfa79fda2aef9c03aa8c17e65aea399c193a503a5c4799dff90"
      "b27fd93ab653ce2f26f04ed5ea96c63d48494f0aa14b580b417b628495265a618712a524"
      "dce0f27c157575d91977b0c496603e71745382eb39e3de3759ec1e77",
      "2bde632fe31e211eca33437566206ea83daa4c5b12aefa42b2f5cde155dfb395c117e29d"
      "6a63c5f70c4a86a39205af99df3c39797cca753e23fc4273e59bd9ff"
      "3e9f2f886d0f897b257b0ded7ffe7134da3f14d2faf7e7f35f3a54b664",
      ""};
  nameCaptureInfo = NULL;
  emlrtNameCaptureMxArrayR2016a(&data[0], 3008U, &nameCaptureInfo);
  return nameCaptureInfo;
}

/*
 * Arguments    : void
 * Return Type  : mxArray *
 */
mxArray *emlrtMexFcnProperties(void)
{
  mxArray *xEntryPoints;
  mxArray *xInputs;
  mxArray *xResult;
  const char_T *propFieldName[9] = {"Version",
                                    "ResolvedFunctions",
                                    "Checksum",
                                    "EntryPoints",
                                    "CoverageInfo",
                                    "IsPolymorphic",
                                    "PropertyList",
                                    "UUID",
                                    "ClassEntryPointIsHandle"};
  const char_T *epFieldName[8] = {
      "Name",     "NumberOfInputs", "NumberOfOutputs", "ConstantInputs",
      "FullPath", "TimeStamp",      "Constructor",     "Visible"};
  xEntryPoints =
      emlrtCreateStructMatrix(1, 1, 8, (const char_T **)&epFieldName[0]);
  xInputs = emlrtCreateLogicalMatrix(1, 1);
  emlrtSetField(xEntryPoints, 0, "Name", emlrtMxCreateString("predictTree"));
  emlrtSetField(xEntryPoints, 0, "NumberOfInputs",
                emlrtMxCreateDoubleScalar(1.0));
  emlrtSetField(xEntryPoints, 0, "NumberOfOutputs",
                emlrtMxCreateDoubleScalar(1.0));
  emlrtSetField(xEntryPoints, 0, "ConstantInputs", xInputs);
  emlrtSetField(xEntryPoints, 0, "FullPath",
                emlrtMxCreateString(
                    "D:\\repository\\deeplearning\\small\\predictTree.m"));
  emlrtSetField(xEntryPoints, 0, "TimeStamp",
                emlrtMxCreateDoubleScalar(739524.58979166672));
  emlrtSetField(xEntryPoints, 0, "Constructor",
                emlrtMxCreateLogicalScalar(false));
  emlrtSetField(xEntryPoints, 0, "Visible", emlrtMxCreateLogicalScalar(true));
  xResult =
      emlrtCreateStructMatrix(1, 1, 9, (const char_T **)&propFieldName[0]);
  emlrtSetField(xResult, 0, "Version",
                emlrtMxCreateString("23.2.0.2515942 (R2023b) Update 7"));
  emlrtSetField(xResult, 0, "ResolvedFunctions",
                (mxArray *)c_emlrtMexFcnResolvedFunctionsI());
  emlrtSetField(xResult, 0, "Checksum",
                emlrtMxCreateString("WkUITSi5UtsZ4Ir8LUGYNH"));
  emlrtSetField(xResult, 0, "EntryPoints", xEntryPoints);
  return xResult;
}

/*
 * File trailer for _coder_predictTree_info.c
 *
 * [EOF]
 */
