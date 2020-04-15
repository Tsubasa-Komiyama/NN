#ifndef __INC_FUNC_H
#define __INC_FUNC_H

#include <stdio.h>

/*!----------------------------------------------------------------------------
 @brif シグモイド関数

  活性化関数としてシグモイド関数を定義. 順伝搬時と逆伝搬時の処理を行う.
 @param [in] array(double*) シグモイド関数を適応させるデータ
 @param [in] size(int) arrayのサイズ
 @param [in] flag(int) 順伝搬か逆伝搬かを判別するための変数
 @param [in, out] matrix(double*) シグモイド関数の微分を格納
 @return double シグモイド関数を適応した入力（0~1)
 @attention
 @par 更新履歴
   - 2020/4/14
     -基本的な機能の実装 (by Tsubasa Komiyama)
   - 2020/4/15
     -仕様および処理を変更 (by Tsubasa Komiyama)
*/

double* Sigmoid(double* array, int size);

/*!----------------------------------------------------------------------------
 @brif ソフトマックス関数

  順伝搬時，逆伝搬時のソフトマックス関数の処理を定義
 @param [in] array(double*) ソフトマックス関数を適応させるデータ
 @param [in] size(int) arrayのサイズ
 @param [in] flag(int) 順伝搬か逆伝搬かを判別するための変数
 @param [in, out] matrix(double*) ソフトマックス関数の微分を格納
 @return double ソフトマックス関数を適応した入力（0~1)
 @attention
 @par 更新履歴
   - 2020/4/14
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

double *Softmax(double* array, int size, int flag, double** matrix);

/*!----------------------------------------------------------------------------
 @brif 平均二乗誤差を求める関数

  損失関数として平均二乗誤差を定義. 順伝搬時と逆伝搬時の処理を行う.
 @param [in] y(double*) 損失を評価するデータ
 @param [in] t(double*) 正解データ
 @param [in] size(int)　yおよびtのサイズ
 @param [in] flag(int)　順伝搬か逆伝搬かを判別するための変数
 @param [in,out] dE_dy(double*)　平均二乗誤差の微分を格納
 @return double yとtの平均二乗誤差
 @attention
 @par 更新履歴
   - 2020/4/14
     -基本的な機能の実装 (by Tsubasa Komiyama)
   - 2020/4/15
     -仕様および処理を変更 (by Tsubasa Komiyama)
*/

double Mean_Square_Error(double* y, double* t, int size, int flag, double* dE_dy);

/*!----------------------------------------------------------------------------
 @brif 順伝搬の処理を行う関数

  順伝搬における入力層，中間層，出力層での処理を定義
 @param [in] nn_param(NN_PARAM) NN_PARAM構造体のデータ
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/15
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

void forward(NN_PARAM nn_param);

/*!----------------------------------------------------------------------------
 @brif 逆伝搬の処理を行う関数

  逆伝搬における入力層，中間層，出力層での処理を定義
 @param [in] nn_param(NN_PARAM) NN_PARAM構造体のデータ
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/15
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

void backward(NN_PARAM nn_param);

/*!----------------------------------------------------------------------------
 @brif 重みの更新を行う関数

  逆伝搬の結果から重みを更新する
 @param [in] nn_param(NN_PARAM) NN_PARAM構造体のデータ
 @param [in] epsilon(double) 学習率
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/15
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

void update_w(NN_PARAM nn_param, double epsilon);

/*!----------------------------------------------------------------------------
 @brif 変数の初期化を行う関数

  変数のメモリ確保を行う
 @param [in] nn_param(NN_PARAM) NN_PARAM構造体のデータ
 @return なし
 @attention
 @par 更新履歴
   - 2020/4/15
     -基本的な機能の実装 (by Tsubasa Komiyama)
*/

void set_variables(NN_PARAM nn_param);



#endif
