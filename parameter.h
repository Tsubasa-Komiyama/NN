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

NN_PARAM set_param(NN_PARAM nn_param)
{
    //num_unitのメモリ確保
    if((nn_param.num_unit = (int*)malloc((nn_param.hidden_layer_size + 1) * sizeof(int))) == NULL){
        exit(-1);
    }

    nn_param.num_unit[0] = nn_param.input_layer_size;
    nn_param.num_unit[nn_param.hidden_layer_size + 1] = nn_param.output_layer_size;

    if((nn_param.act = (double* (**)(double*, int, int, double**))malloc((nn_param.hidden_layer_size + 2) * sizeof(double (**)(double*, double*, int, int, double*)))) == NULL){
        exit(-1);
    }

    nn_param.act[0] = NULL;
    nn_param.loss = NULL;

    return nn_param;
}

//変数
double **data = NULL;        //入力データ
int *size = NULL;           //各層の素子数
double ***w = NULL;         //重み
double **layer_in = NULL;   //各層の入力
double **layer_out = NULL;  //各層の出力
double *out = NULL;         //出力層の出力
double *t = NULL;           //正解データ

double ***dE_dw = NULL;         //
double ***dE_dw_t = NULL;   //
double **dE_da = NULL;          //

#endif
