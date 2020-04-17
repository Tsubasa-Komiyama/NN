#include <stdlib.h>
#ifndef __INC_PARAMETER_H
#define __INC_PARAMETER_H

//構造体
typedef struct {
    int input_layer_size;   //入力層のサイズ
    int hidden_layer_size;  //中間層のサイズ
    int output_layer_size;  //出力層のサイズ
    int *num_unit;    //中間層の素子数
    double *(**act)(double *array, int size, int flag, double **matrix);     //各層の活性化関数
    double (*loss)(double *y, double *t, int size, int flag, double *dE_dy);
} NN_PARAM;

//変数
double **train_data;        //入力データ
int *size;                  //各層の素子数
double ***w;                //重み
double **layer_in;          //各層の入力
double **layer_out;         //各層の出力
double **out;                //出力層の出力
double **t;                  //正解データ
double **unlearn_data;      //未学習データ

double ***dE_dw;            //
double ***dE_dw_t;          //
double **dE_da;             //

#endif
