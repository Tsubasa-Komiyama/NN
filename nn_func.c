#include "nn_func.h"
#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//シグモイド関数
double *Sigmoid(double *array, int size, int flag, double **matrix)
{
    double *y = NULL; //出力する配列

    //動的メモリ確保
    if((y = (double*)malloc((size + 1) * sizeof(double))) == NULL){
        return NULL;
    }

    y[0] = 1.0;   //一個目の要素を1にする

    //各要素について値の計算を行う
    for(int i = 1; i <= size; i++){
        y[i] = 1.0 / (1.0 + exp(-array[i]));
    }

    //逆伝搬時, 微分を計算
    if(flag){
        //初期化
        for(int i = 1; i <= size; i++){
            for(int j = 1; j <= size; j++){
                matrix[i][j] = 0.0;
            }
        }

        //微分を計算
        for(int i = 0; i <= size; i++){
            matrix[i][i] = y[i] * (1.0 - y[i]);
        }

        return NULL;
    }

    return y;
}


//ソフトマックス関数
double *Softmax(double* array, int size, int flag, double** matrix)
{
    double *y = NULL;
    double sum_exp = 0.0;
    double max_a = array[1];

    for(int i = 2; i <= size; i++){
        if(array[i] > max_a){
            max_a = array[i];
        }
    }

    //動的メモリ確保
    if((y = (double*)malloc((size + 1) * sizeof(double))) == NULL){
        return NULL;
    }

    y[0] = 1.0;

    for(int i = 1; i <= size; i++){
        y[i]  = exp(array[i] - max_a);
        sum_exp += y[i];
    }

    for(int i = 1; i <= size; i++){
        y[i] /= sum_exp;
    }

    //逆伝搬時
    if(flag){
        for(int i = 1; i <= size; i++){
            for(int j = 1; j <= i; j++){
                matrix[i][j] = -y[i] * y[j];
                matrix[j][i] = -y[i] * y[j];
            }

            matrix[i][i] = y[i] * (1.0 - y[i]);
        }

        return NULL;
    }

    return y;
}


//平均二乗誤差
double Mean_Square_Error(double *y, double *t, int size, int flag, double *dE_dy)
{
    double x;   //y-t
    double e = 0.0; //sum((y-t)^2)

    //逆伝搬時，微分を計算
    if(flag){
        for(int i = 1; i <= size; i++){
            dE_dy[i] = 2.0 * (y[i] - t[i]) / size;
        }

        return 0.0;
    }

    //二乗誤差を計算
    for(int i = 1; i <= size; i++){
        x = y[i] - t[i];
        e += x*x;
    }

    return e / (double)size;
}


//順伝搬
void forward(NN_PARAM nn_param, double *data, double ***w, int *size, double **layer_in, double **layer_out, double *out)
{
    //printf("out : %p\n", out);
    int i, j, k;     //制御変数
    double tmp;     //
    double *layer_out_p;
    double *out_p;

    //入力層
    layer_out[0] = data;    //入力層では入力データをそのまま出力する

    //中間層
    for(i = 1; i <= nn_param.hidden_layer_size; i++){       //i：中間層のインデックス
        int prev_layer_size = size[i-1];
        int curr_layer_size = size[i];

        for(j = 1; j <= curr_layer_size; j++){    //j：中間層第i+1層の素子のインデックス
            tmp = w[i-1][0][j];   //バイアス

            for(k = 1; k <= prev_layer_size; k++){    //k：中間層第i層の素子のインデックス
                //前の層の出力をすべて足す
                tmp += w[i-1][k][j] * layer_out[i-1][k];
            }

            layer_in[i][j] = tmp;     //次の層の入力に代入
        }

        layer_out_p = nn_param.act[i](layer_in[i], size[i], 0, NULL);

        //入力をシグモイド関数で活性化し，出力に入れる
        for(j = 0; j <= curr_layer_size; j++){    //j：中間層第i+1層の素子のインデックス
            layer_out[i][j] = *layer_out_p;
            layer_out_p++;
        }
    }

    //出力層
    int prev_layer_size = nn_param.hidden_layer_size;
    //printf("prev_layer_size : %d\n", prev_layer_size);

    for(i = 1; i <= nn_param.output_layer_size; i++){   //i：出力層の素子のインデックス
        tmp = w[nn_param.hidden_layer_size][0][i];  //バイアス

        for(j = 0; j <= prev_layer_size; j++){    //j：中間層の最後の層の素子のインデックス
            tmp += w[nn_param.hidden_layer_size][j][i] * layer_out[nn_param.hidden_layer_size][j];
        }

        layer_in[nn_param.hidden_layer_size + 1][i] = tmp;
    }

    out_p = nn_param.act[nn_param.hidden_layer_size + 1](layer_in[nn_param.hidden_layer_size + 1], nn_param.output_layer_size, 0, NULL);
    for(i = 0; i <= nn_param.output_layer_size; i++){
        out[i] = *out_p;
        out_p++;
    }
}


//逆伝搬
void backward(NN_PARAM nn_param, double ***w, int *size, double **layer_in, double **layer_out, double *out, double *t, double ***dE_dw, double ***dE_dw_t, double **dE_da)
{
    int i, j, k;     //制御変数
    double *dE_dy;      //出力層での損失関数の微分
    double **dy_da;
    double **dz_da;
    double tmp;

    //dE_dyの初期化
    if((dE_dy = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL){
        exit(-1);
    }
    for(i = 0; i <= nn_param.output_layer_size; i++){
        dE_dy[i] = 0.0;
    }

    //dE_dyを計算
    nn_param.loss(out, t, nn_param.output_layer_size, 1, dE_dy);

    //dy_daを初期化，メモリ確保
    if((dy_da = (double**)malloc((nn_param.output_layer_size + 1) * sizeof(double*))) == NULL){
        exit(-1);
    }

    for(i = 0; i <= nn_param.output_layer_size; i++){
        if((dy_da[i] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    //出力層
    //dy_daを計算
    nn_param.act[nn_param.hidden_layer_size + 1](layer_in[nn_param.hidden_layer_size + 1], nn_param.output_layer_size, 1, dy_da);

    for(i = 1; i <= nn_param.output_layer_size; i++){
        tmp = 0.0;

        for(j = 1; j <= nn_param.output_layer_size; j++){
            tmp += dE_dy[j] * dy_da[j][i];
        }

        dE_da[nn_param.hidden_layer_size + 1][i] = tmp;
    }

    //メモリ解放
    free(dE_dy);

    for(i = 0; i <= nn_param.output_layer_size; i++){
        free(dy_da[i]);
    }
    free(dy_da);

    //中間層
    for(i = nn_param.hidden_layer_size; i >= 1; i--){    //i：中間層のインデックス
        int curr_layer_size = size[i];
        int next_layer_size = size[i+1];
        double z = 0.0;     //layer_out[i][j]

        //dE_dwの計算
        for(j = 0; j <= curr_layer_size; j++){    //j：中間層第i層の素子のインデックス
            z = layer_out[i][j];

            for(k = 0; k <= next_layer_size; k++){    //k：中間層第i+1層の素子のインデックス
                dE_dw[i][j][k] = z * dE_da[i+1][k];
                dE_dw_t[i][j][k] += dE_dw[i][j][k];
            }
        }

        //dz_daの初期化
        if((dz_da = (double**)malloc((curr_layer_size + 1) * sizeof(double*))) == NULL){
            exit(-1);
        }

        for(j = 0; j <= curr_layer_size; j++){
            if((dz_da[j] = (double*)malloc((curr_layer_size + 1) * sizeof(double))) == NULL){
                exit(-1);
            }

            for(k = 0; k <= curr_layer_size; k++){
                dz_da[j][k] = 0.0;
            }
        }

        //dz_daの計算
        nn_param.act[i](layer_in[i], curr_layer_size, 1, dz_da);

        //dE_daの計算
        for(j = 1; j <= curr_layer_size; j++){
            tmp = 0.0;

            for(k = 1; k <= next_layer_size; k++){
                tmp += w[i][j][k] * dE_da[i+1][k];
            }

            dE_da[i][j] = dz_da[j][j] * tmp;
        }

        //メモリの解放
        for(j = 0; j <= curr_layer_size; j++){
            free(dz_da[j]);
        }

        free(dz_da);
    }

    //入力層
    //dE_dwの計算
    int next_layer_size = size[1];

    for(i = 0; i <= nn_param.input_layer_size; i++){
        for(j = 0; j <= next_layer_size; j++){
            dE_dw[0][i][j] = layer_out[0][i] * dE_da[1][j];
            dE_dw_t[0][i][j] += dE_dw[0][i][j];
        }
    }

}


//重みの更新
void update_w(NN_PARAM nn_param, double epsilon, double ***w, int *size, double ***dE_dw)
{
    for(int i = 0; i <= nn_param.hidden_layer_size; i++){
        int curr_layer_size = size[i];
        int next_layer_size = size[i+1];

        for(int j = 1; j <= curr_layer_size; j++){
            for(int k = 0; k <= next_layer_size; k++){
                w[i][j][k] -= epsilon * dE_dw[i][j][k];
            }
        }
    }
}


void batch_update_w(NN_PARAM nn_param, double epsilon, double ***w, int *size, double ***dE_dw_t, int batch_size)
{
    for(int i = 0; i <= nn_param.hidden_layer_size; i++){
        int curr_layer_size = size[i];
        int next_layer_size = size[i+1];

        for(int j = 1; j <= curr_layer_size; j++){
            for(int k = 0; k <= next_layer_size; k++){
                w[i][j][k] -= epsilon * dE_dw_t[i][j][k] / batch_size;
            }
        }
    }
}

//構造体の設定
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
