#include "Func.h"
#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//シグモイド関数
double *Sigmoid(double *array, int size, int flag, double *matrix)
{
    double *out = NULL; //出力する配列

    //動的メモリ確保
    if((out = (double*)malloc((size + 1) * sizeof(double))) = NULL){
        return NULL;
    }

    out[0] = 0.0;   //一個目の要素を0にする

    //各要素について値の計算を行う
    for(int i = 1; i <= size; i++){
        out[i] = 1.0 / (1.0 + exp(-array[i]));
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
            matrix[i][i] = out[i] * (1.0 - out[i]);
        }

        return NULL;
    }

    return out;
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
            dE_dy[i] = 2.0 * (y[i] - t[i]) / double(size);
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
void forward(NN_PARAM nn_param)
{
    int i, j, k;     //制御変数
    double tmp;     //

    //入力層
    layer_out[0] = data;

    //中間層
    for(i = 0; i < nn_param.hidden_layer_size; i++){
        int prev_layer_size = size[i];
        int curr_layer_size = size[i+1]:

        for(j = 1; j <= curr_layer_size; j++){
            tmp = w[i][0][j];   //バイアス

            for(k = 1; k <= prev_layer_size; k++){
                tmp += w[i][k][j] * layer_out[i][k];
            }

            layer_in[i+1][j] = tmp;     //次の層の入力に代入
        }

        free(layer_out[i+1]);

        later_out[i+1] = nn_param.act[i+1](layer_in[i+1], size[i+1], 0, NULL);
    }

    //出力層
    int prev_layer_size = size[i];

    for(i = 1; i <= nn_param.output_layer_size; i++){
        tmp = w[nn_param.hidden_layer_size][0][i];

        for(j = 0; j <= prev_layer_size; j++){
            tmp += w[nn_param.hidden_layer_size][j][i] * layer_out[nn_param.hidden_layer_size][j];
        }

        layer_in[nn_param.hidden_layer_size + 1][i] = tmp;
    }

    free(out);

    out = nn_param.act[nn_param.hidden_layer_size + 1](layer_in[nn_param.hidden_layer_size + 1], nn_param.output_layer_size, 0, NULL);
}


//逆伝搬
void backward(NN_PARAM nn_param)
{
    int i, j, k;     //制御変数
    double *dE_dy;      //出力層での損失関数の微分
    double **dy_da;     //
    double **dz_da;     //
    double tmp;

    //dE_dyの初期化
    if((dE_dy = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) = NULL){
        exit(-1);
    }
    for(i = 0; i <= nn_param.output_layer_size; i++){
        dE_dy[i] = 0.0;
    }

    //dE_dyを計算
    nn_param.loss(out, t, nn_param.output_layer_size, 1, dE_dy);

    //dy_daを初期化，メモリ確保
    if((dy_da = (double**)malloc((nn_param.output_layer_size + 1) * sizeof(double*))) = NULL){
        exit(-1);
    }

    for(i = 1; i <= nn_param.output_layer_size; i++){
        if((dy_da[i] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) = NULL){
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

    for(i = 1; i <= nn_param.output_layer_size; i++){
        free(dy_da[i]);
    }

    free(dy_da);

    //中間層
    for(i = nn_param.hidden_layer_size; i >= 1; i--){
        int curr_layer_size = size[i];
        int next_layer_size = size[i+1];
        double z = 0.0;     //layer_out[i][j]

        //dE_dwの計算
        for(j = 0; j <= curr_layer_size; j++){
            z = layer_out[i][j];

            for(k = 0; k <= next_layer_size; k++){
                dE_dw[i][j][k] = z * dE_da[i+1][k];
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
        }
    }

}


//重みの更新
void update_w(NN_PARAM nn_param, double epsilon)
{
    for(int i = 0; i <= nn_param.hidden_layer_size; i++){
        int curr_layer_size = size[i];
        int next_layer_size = size[i+1];

        for(int j = 1; j <= curr_layer_size; j++){
            for(int k = 0; k <= next_layer_size; k++){
                w[i][j][k] -= epsilon * dE_dw_t[i][j][k];
            }
        }
    }
}


//変数の設定
void set_variables(NN_PARAM nn_param)
{
    if((data = (double*)malloc((nn_param.input_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    //size
    nn_param.num_unit[0] = nn_param.input_layer_size;
    size = nn_param.N_unit;

    //w
    if((w = (double***)malloc((nn_param.hidden_layer_size + 1) * sizeof(double**))) == NULL) {
        exit(-1);
    }

    srand((unsigned int)time(NULL));

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        int curr_layer_size = size[i];
        int next_layer_size = size[i + 1];

        if((w[i] = (double**)malloc((d_max + 1) * sizeof(double*))) == NULL) {
            exit(-1);
        }

        for(i = 0; i <= curr_layer_size; i++) {
            if((w[l][i] = (double*)malloc((next_layer_size + 1) * sizeof(double))) == NULL) {
                exit(-1);
            }

            for(j = 0; j <= d_next_layer; j++) {
                w[l][i][j] = ((double)rand()/RAND_MAX) * 2 - 1;  //乱数でaを初期化
            }
        }
    }

    //layer_out
    if((layer_in = (double**)malloc((nn_param.hidden_layer_size + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        int curr_layer_size = size[i];

        if((layer_out[i] = (double*)malloc((curr_layer_size + 1) * sizeof(double))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= curr_layer_size; j++) {
            layer_out[i][j] = 0.0;
        }
    }

    //layer_in
    if((layer_in = (double**)malloc((nn_param.hidden_layer_size + 2) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        if((layer_in[i] = (double*)malloc((size[i] + 1) * sizeof(double))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= size[i]; j++) {
            layer_in[i][j] = 0.0;
        }
    }

    if((layer_in[nn_param.hidden_layer_size + 1] = (double*)malloc((nn_param.out_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.out_layer_size; i++) {
        layer_in[nn_param.out_layer_size + 1][i] = 0.0;
    }

    //out
    if((out = (double*)malloc((nn_param.out_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.out_layer_size; i++) {
        out[i] = 0.0;
    }

    // Setup t
    if((t = (double*)malloc((nn_param.out_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.out_layer_size; i++) {
        t[i] = 0.0;
    }

    //dE_dw & dE_dw_t
    if((dE_dw = (double***)malloc((nn_param.hidden_layer_size + 1) * sizeof(double**))) == NULL || (dE_dw_t = (double***)malloc((nn_param.hidden_layer_size + 1) * sizeof(double**))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        if((dE_dw[i] = (double**)malloc((size[i] + 1) * sizeof(double*))) == NULL || (dE_dw_t[i] = (double**)malloc((size[i] + 1) * sizeof(double*))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= size[i]; j++) {
            if((dE_dw[i][j] = (double*)malloc((size[i+1] + 1) * sizeof(double))) == NULL || (dE_dw_t[i][j] = (double*)malloc((size[i+1] + 1) * sizeof(double))) == NULL) {
                exit(-1);
            }

            for(k = 0; k <= size[i+1]; k++) {
                dE_dw[i][j][k] = 0.0;
                dE_dw_t[i][j][k] = 0.0;
            }
        }
    }

    //dE_da
    if((dE_da = (double**)malloc((nn_param.hidden_layer_size + 2) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        if((dE_da[i] = (double*)malloc((size[i] + 1) * sizeof(double))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= nn_param.hidden_layer_size; j++) {
            dE_da[i][j] = 0.0;
        }
    }

    if((dE_da[nn_param.hidden_layer_size + 1] = (double*)malloc((nn_param.out_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.out_layer_size; i++) {
        dE_da[nn_param.hidden_layer_size + 1][i] = 0.0;
    }
}
