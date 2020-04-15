#include <stdio.h>
#include "nn_func.h"
#include "parameter.h"
#include <assert.h>

#define N 100     //試行回数の最大値

int main(void){
    int i, j;  //制御変数
    int train_num;  //訓練データの数
    NN_PARAM nn_param;
    double unit_N;
    FILE *fp;
    double Loss;    //損失関数
    double epsilon; //学習率

    //NN_PARAMの要素の定義
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

    nn_param = set_param(nn_param);

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.num_unit[i] = unit_N;
    }

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.act[i] = Sigmoid;
    }

    nn_param.act[nn_param.hidden_layer_size + 1] = Softmax;
    nn_param.loss = Mean_Square_Error;

    set_variables(nn_param);

    //ファイルオープン
	if ((fp = fopen("data.txt", "r")) == NULL) {
        printf("ファイルを開けませんでした．\n");
        return -1;
	}

	//データ読み込み
    i = 0;
	while (fgetc(fp) != '\n' && !feof(fp));
    while(fscanf(fp, "%lf %lf %lf %lf", &data[i][0], &data[i][1], &data[i][2], &t[i]) == EOF){
        i++;
    }

    train_num = i;

    fclose(fp);

    for(i = 0; i < N; i++){
        for(j = 0; j < train_num; j++){
            forward(nn_param, data[i]);

            Loss = nn_param.loss(out, t, nn_param.output_layer_size, 0, NULL);

            printf("%d : %lf\n", i, Loss);

            backward(nn_param);
            update_w(nn_param, epsilon);
        }
    }

    return 0;
}
