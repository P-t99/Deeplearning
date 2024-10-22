/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * File: nczallcode.h
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

#ifndef nczallcode_h_
#define nczallcode_h_
#ifndef nczallcode_COMMON_INCLUDES_
#define nczallcode_COMMON_INCLUDES_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_nonfinite.h"
#include "math.h"
#endif                                 /* nczallcode_COMMON_INCLUDES_ */

#include "nczallcode_types.h"
#include "rtGetNaN.h"

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

/* Block states (default storage) for system '<Root>' */
typedef struct {
  ClassificationTree_nczallcode_T treeModel;/* '<Root>/MATLAB Function' */
  real_T Awake_sleep_average[48];      /* '<Root>/Data Store Memory22' */
  real_T onbednum;                     /* '<Root>/Data Store Memory23' */
  real_T BREATH_COUNT;                 /* '<Root>/Data Store Memory36' */
  real_T TRIGGER_COUNT;                /* '<Root>/Data Store Memory37' */
  real_T previousFramesArray[1536];    /* '<Root>/Data Store Memory41' */
  real_T BREATH_COUNTArray[16];        /* '<Root>/Data Store Memory42' */
  real_T conditionTriggerHistoryArray[80];/* '<Root>/Data Store Memory43' */
  real_T TRIGGER_COUNTArray[16];       /* '<Root>/Data Store Memory44' */
  real_T dominantFrequencyArray[16];   /* '<Root>/Data Store Memory45' */
  real32_T G_INBED_TIM;                /* '<Root>/Data Store Memory' */
  real32_T BREATH_RATE;                /* '<Root>/Data Store Memory10' */
  real32_T MIN_SEND_N;                 /* '<Root>/Data Store Memory11' */
  real32_T STROKE_RISK_MIN;            /* '<Root>/Data Store Memory12' */
  real32_T BREATH_MIN;                 /* '<Root>/Data Store Memory13' */
  real32_T BATCH_FRA_MEAN[720];        /* '<Root>/Data Store Memory14' */
  real32_T TWENTY_SEC;                 /* '<Root>/Data Store Memory15' */
  real32_T STATE_SLEEP[48];            /* '<Root>/Data Store Memory16' */
  real32_T FRAME_DIFF[16];             /* '<Root>/Data Store Memory17' */
  real32_T ITER;                       /* '<Root>/Data Store Memory18' */
  real32_T VARN;                       /* '<Root>/Data Store Memory19' */
  real32_T G_STATE_INBED;              /* '<Root>/Data Store Memory2' */
  real32_T L_R_M_FREATUE[180];         /* '<Root>/Data Store Memory20' */
  real32_T LEFT_RIGHT_MOVE_SUM[4];     /* '<Root>/Data Store Memory21' */
  real32_T R_ALARM[180];               /* '<Root>/Data Store Memory24' */
  real32_T BREATH_SW;                  /* '<Root>/Data Store Memory26' */
  real32_T PRESSURE_MEAN;              /* '<Root>/Data Store Memory27' */
  real32_T ALARM_NUM;                  /* '<Root>/Data Store Memory3' */
  real32_T BODY_MOVE_NUM;              /* '<Root>/Data Store Memory30' */
  real32_T ALL_MOVE_NUM;               /* '<Root>/Data Store Memory32' */
  real32_T ONBED_SW;                   /* '<Root>/Data Store Memory33' */
  real32_T inBedTimer;                 /* '<Root>/Data Store Memory34' */
  real32_T PRO_FRAME[160];             /* '<Root>/Data Store Memory4' */
  real32_T NCZ_TIMES;                  /* '<Root>/Data Store Memory5' */
  real32_T realArr[4000];              /* '<Root>/Data Store Memory6' */
  real32_T ALL_DIFF_IMG[24];           /* '<Root>/Data Store Memory7' */
  real32_T SEND_DATA_N;                /* '<Root>/Data Store Memory8' */
  real32_T STROKE_RISK;                /* '<Root>/Data Store Memory9' */
  boolean_T treeModel_not_empty;       /* '<Root>/MATLAB Function' */
} DW_nczallcode_T;

/* External inputs (root inport signals with default storage) */
typedef struct {
  real32_T frameData[160];             /* '<Root>/frameData' */
  real32_T tim;                        /* '<Root>/tim ' */
} ExtU_nczallcode_T;

/* External outputs (root outports fed by signals with default storage) */
typedef struct {
  real32_T bodyMovementData[24];       /* '<Root>/bodyMovementData' */
  real32_T rate;                       /* '<Root>/rate' */
  real32_T strokerisk;                 /* '<Root>/strokerisk' */
  real32_T stateInBbed;                /* '<Root>/stateInBbed' */
  real32_T inBedtime;                  /* '<Root>/inBedtime' */
  real32_T rateMin;                    /* '<Root>/rateMin' */
  real32_T strokeriskMin;              /* '<Root>/strokeriskMin' */
  real32_T statesleep[48];             /* '<Root>/statesleep' */
  real32_T timt;                       /* '<Root>/timt' */
  real32_T meansn;                     /* '<Root>/meansn' */
  real32_T frame_img[160];             /* '<Root>/frame_img' */
  real32_T pos;                        /* '<Root>/pos' */
} ExtY_nczallcode_T;

/* Real-time Model Data Structure */
struct tag_RTM_nczallcode_T {
  const char_T * volatile errorStatus;
};

/* Block states (default storage) */
extern DW_nczallcode_T nczallcode_DW;

/* External inputs (root inport signals with default storage) */
extern ExtU_nczallcode_T nczallcode_U;

/* External outputs (root outports fed by signals with default storage) */
extern ExtY_nczallcode_T nczallcode_Y;

/* Model entry point functions */
extern void nczallcode_initialize(void);
extern void nczallcode_step(void);
extern void nczallcode_terminate(void);

/* Real-time Model object */
extern RT_MODEL_nczallcode_T *const nczallcode_M;

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'nczallcode'
 * '<S1>'   : 'nczallcode/MATLAB Function'
 */
#endif                                 /* nczallcode_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
