#include <stdio.h>
#include "Func.h"
#include "parameter.h"

int main(void){
    int i, j, k;  //制御変数
    NN_PARAM nn_param;
    double unit_N;

    //NN_PARAMの要素の定義
    printf("中間層の層数を入力してください：\n");
    scanf("%lf", &nn_param.hidden_layer_size);

    printf("中間層の素子数を入力してください：\n");
    scanf("%lf", &unit_N);

    printf("出力層の素子数を入力してください：\n");
    scanf("%lf", &nn_param.out_layer_size);

    nn_param.input_layer_size = data????;

    nn_param = set_param(nn_param);

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.num_unit[i] = unit_N;
    }

    for(i = 1; i <= nn_param.hidden_layer_size; i++){
        nn_param.act[i] = Sigmoid;
    }

    nn_param.act[nn_param.hidden_layer_size + 1] = softmax;
    nn_param.loss = Mean_Square_Error;

    set_variables(nn_param);

    
}
