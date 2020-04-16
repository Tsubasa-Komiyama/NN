#include <stdio.h>
#include "nn_func.h"
#include "parameter.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>

#define KEY_SIZE 2  //keyの要素数
#define N 10000       //試行回数の最大値
#define LOSS_MIN 0.0000001 //損失の閾値

int main(void){
    int i, j, k;        //制御変数
    int key;  //対話用選択肢
    int train_num;      //訓練データの数
    NN_PARAM nn_param;  //構造体
    double unit_N;      //素子数
    FILE *fp;
    double Loss_batch;
    double Loss_seq;        //損失関数
    double epsilon;     //学習率
    int batch_count;    //一括学習回数
    int seq_count;      //逐次学習回数
    int data_num = 6;   //データの数


    /**************************層数・素子数の設定*****************************/
    printf("中間層の層数を入力してください：\n");
    scanf("%d", &nn_param.hidden_layer_size);

    printf("中間層の素子数を入力してください：\n");
    scanf("%lf", &unit_N);

    printf("入力層の層数を入力してください：\n");
    scanf("%d", &nn_param.input_layer_size);
    assert(nn_param.input_layer_size >= 3);

    printf("出力層の素子数を入力してください：\n");
    scanf("%d", &nn_param.output_layer_size);

    printf("学習率を入力してください：\n");
    scanf("%lf", &epsilon);


    /**************************各種パラメータの設定*****************************/
    nn_param = set_param(nn_param);

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.num_unit[i] = unit_N;
    }

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.act[i] = Sigmoid;
    }

    nn_param.act[nn_param.hidden_layer_size + 1] = Softmax;
    nn_param.loss = Mean_Square_Error;

    double **train_data = NULL;        //入力データ
    int *size = NULL;           //各層の素子数
    double ***w = NULL;         //重み
    double **layer_in = NULL;   //各層の入力
    double **layer_out = NULL;  //各層の出力
    double **out = NULL;         //出力層の出力
    double **t = NULL;           //正解データ
    double **unlearn_data;      //未学習データ

    double ***dE_dw = NULL;         //
    double ***dE_dw_t = NULL;   //
    double **dE_da = NULL;          //

    //train_data
    if((train_data = (double**)malloc((data_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= data_num; i++){
        if((train_data[i] = (double*)malloc((nn_param.input_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    //size
    nn_param.num_unit[0] = nn_param.input_layer_size;
    size = nn_param.num_unit;

    //w
    if((w = (double***)malloc((nn_param.hidden_layer_size + 1) * sizeof(double**))) == NULL) {
        exit(-1);
    }

    srand((unsigned int)time(NULL));

    for(i = 0; i <= nn_param.hidden_layer_size; i++) {
        if((w[i] = (double**)malloc((size[i] + 1) * sizeof(double*))) == NULL) {
            exit(-1);
        }

        for(j = 0; j <= size[i]; j++) {
            if((w[i][j] = (double*)malloc((size[i+1] + 1) * sizeof(double))) == NULL) {
                exit(-1);
            }

            for(k = 0; k <= size[i+1]; k++) {
                w[i][j][k] = ((double)rand()/RAND_MAX) * 2 - 1;  //乱数でaを初期化
            }
        }
    }

    //layer_out
    if((layer_out = (double**)malloc((nn_param.hidden_layer_size + 1) * sizeof(double*))) == NULL) {
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

    if((layer_in[nn_param.hidden_layer_size + 1] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.output_layer_size; i++) {
        layer_in[nn_param.output_layer_size + 1][i] = 0.0;
    }

    //out
    if((out = (double**)malloc((data_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= data_num; i++){
        if((out[i] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    for(i = 0; i <= data_num; i++){
        for(j = 0; j <= nn_param.output_layer_size; j++){
            out[i][j] = 0.0;
        }
    }

    //t
    if((t = (double**)malloc((data_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= data_num; i++){
        if((t[i] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
    }

    for(i = 0; i <= data_num; i++){
        for(j = 0; j <= nn_param.output_layer_size; j++){
            t[i][j] = 0.0;
        }
    }

    //unlearn_data
    if((unlearn_data = (double**)malloc((data_num + 1) * sizeof(double*))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= data_num; i++){
        if((unlearn_data[i] = (double*)malloc((nn_param.input_layer_size + 1) * sizeof(double))) == NULL){
            exit(-1);
        }
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

    if((dE_da[nn_param.hidden_layer_size + 1] = (double*)malloc((nn_param.output_layer_size + 1) * sizeof(double))) == NULL) {
        exit(-1);
    }

    for(i = 0; i <= nn_param.output_layer_size; i++) {
        dE_da[nn_param.hidden_layer_size + 1][i] = 0.0;
    }


    /**************************教師データ・未学習データの読み込み*****************************/

    //教師データ
    //ファイルオープン
	if ((fp = fopen("train_data.txt", "r")) == NULL) {
        printf("ファイルを開けませんでした．\n");
        return -1;
	}
	//データ読み込み
    printf("教師データ\n");
    i = 0;
	while ((fscanf(fp, "%lf %lf %lf %lf", &train_data[i][0], &train_data[i][1], &train_data[i][2], &t[i][0])) != EOF) {
		printf("%.1lf %.1lf %.1lf %.1lf\n", train_data[i][0], train_data[i][1], train_data[i][2], t[i][0]);
		i++;
	}
    train_num = i;
    fclose(fp);

    //未学習データ
    //ファイルオープン
	if ((fp = fopen("unlearn_data.txt", "r")) == NULL) {
        printf("ファイルを開けませんでした．\n");
        return -1;
	}
	//データ読み込み
    printf("未学習データ\n");
    i = 0;
	while ((fscanf(fp, "%lf %lf %lf", &unlearn_data[i][0], &unlearn_data[i][1], &unlearn_data[i][2])) != EOF) {
		printf("%.1lf %.1lf %.1lf\n", unlearn_data[i][0], unlearn_data[i][1], unlearn_data[i][2]);
		i++;
	}
    fclose(fp);

    /**************************システム*****************************/

    printf("**************************************************\n");
    printf("ニューラルネットワーク\n");
    printf("**************************************************\n");


    while(1){

      printf("[a] 一括更新学習法\n");
      printf("[b] 逐次更新学習法\n");
      printf("[c] 学習済みニューロンのテスト\n");
      printf("[ESC] プログラム終了\n");

      printf("**************************************************\n");
      printf("キーを入力して機能を選択してください：\n");
      key = getch();
      printf("**************************************************\n");

      switch (key) {
        case 'a':
        do{
            batch_count = 0;

            //教師データについて一つずつ順伝搬・逆伝搬を行い，重みの更新はエポックごとに行う
            Loss_batch = 0.0;
            for(i = 0; i < nn_param.input_layer_size; i++){
                //順伝搬
                forward(nn_param, train_data[i], w, size, layer_in, layer_out, out[i]);
                Loss_batch += nn_param.loss(out[i], t[i], nn_param.output_layer_size, 0, NULL) / train_num;
                //逆伝搬
                backward(nn_param, w, size, layer_in, layer_out, out[i], t[i], dE_dw, dE_dw_t, dE_da);
            }

            //重みの更新
            batch_update_w(nn_param, epsilon, w, size, dE_dw_t, train_num);

            //カウント
            batch_count++;

            printf("%d : %lf\n", batch_count, Loss_batch);
        }while(abs(Loss_batch) > LOSS_MIN && batch_count < N);

          if(abs(Loss_batch) < LOSS_MIN){
            printf("勾配の大きさが一定値を下回りました。\n");
            //printf("パラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", batch_count);
        }else if(seq_count > N){
            printf("繰り返し回数が一定数を超えました。\n");
            //printf("パラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", batch_count);
          }else{
            //printf("その時のパラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", seq_count);
          }
          printf("**************************************************\n");
          break;

        case 'b':
        do{
            seq_count = 0;

            //教師データについて一つずつ順伝搬・逆伝搬・重みの更新を行う
            for(i = 0; i < nn_param.input_layer_size; i++){
                //順伝搬
                forward(nn_param, train_data[i], w, size, layer_in, layer_out, out[i]);
                Loss_seq = nn_param.loss(out[i], t[i], nn_param.output_layer_size, 0, NULL);
                //逆伝搬
                backward(nn_param, w, size, layer_in, layer_out, out[i], t[i], dE_dw, dE_dw_t, dE_da);
                //重みの更新
                update_w(nn_param, epsilon, w, size, dE_dw_t);
            }
            //カウント
            seq_count++;

            printf("%d : %lf\n", seq_count, Loss_seq);
        }while(abs(Loss_seq) > LOSS_MIN && seq_count < N);

          if(abs(Loss_seq) < LOSS_MIN){
            printf("勾配の大きさが一定値を下回りました。\n");
            //printf("パラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", seq_count);
        }else if(seq_count > N){
            printf("繰り返し回数が一定数を超えました。\n");
            //printf("パラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", seq_count);
          }else{
            //printf("その時のパラメータの値は%.4lf\n", a);
            printf("繰り返し回数は%d\n", seq_count);
          }
          printf("**************************************************\n");
          break;

        case 'c':

         printf("**************************************************\n");
         break;

        case 0x1b:
            printf("プログラムを終了します.\n");
            return -1;
      }
    }





    /*
    data[0][0] = 0.0;
    data[0][1] = 0.0;
    data[0][2] = 0.0;
    data[1][0] = 1.0;
    data[1][1] = 0.0;
    data[1][2] = 1.0;
    data[2][0] = 1.0;
    data[2][1] = 1.0;
    data[2][2] = 1.0;
    data[3][0] = 1.0;
    data[3][1] = 1.0;
    data[3][2] = 0.0;
    data[4][0] = 1.0;
    data[4][1] = 0.0;
    data[4][2] = 0.0;
    data[5][0] = 0.0;
    data[5][1] = 0.0;
    data[5][2] = 1.0;
    */

    /*
    t[0] = 0.0;
    t[1] = 0.0;
    t[2] = 1.0;
    t[3] = 0.0;
    t[4] = 1.0;
    t[5] = 1.0;
    */




    //メモリ開放
    for(i = 0; i <= nn_param.input_layer_size; i++){
        free(train_data[i]);
    }
    free(train_data);
    for(i = 0; i <= nn_param.input_layer_size; i++){
        free(unlearn_data[i]);
    }
    free(unlearn_data);
    for(i = 0; i <= nn_param.hidden_layer_size; i++){
        for(j = 0; j <= size[i]; j++){
            free(w[i][j]);
        }

        free(w[i]);
    }
    free(w);
    for(i = 0; i <= nn_param.hidden_layer_size; i++){
        free(layer_out[i]);
    }
    free(layer_out);
    for(i = 0; i <= nn_param.hidden_layer_size; i++){
        free(layer_in[i]);
    }
    free(layer_in);
    free(out);
    free(t);
    for(i = 0; i <= nn_param.hidden_layer_size; i++){
        for(j = 0; j <= size[i]; j++){
            free(dE_dw[i][j]);
            free(dE_dw_t[i][j]);
        }

        free(dE_dw[i]);
        free(dE_dw_t[i]);
    }
    free(dE_dw);
    free(dE_dw_t);
    for(i = 0; i <= nn_param.hidden_layer_size; i++){
        free(dE_da[i]);
    }
    free(dE_da);



    return 0;
}
