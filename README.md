# Name

Three input one output Neural Network
3入力１出力ニューラルネットワーク

nn_func.cにある関数についてはn_func.hをお読みください．

# Usage

まず最初に，中間層の層数，素子数，入力層の素子数，出力層の素子数を入力してください．
次にa, b, c, d, escのキーのうち，どれかを押して行う処理を選択してください．
aは一括学習，bは逐次学習を行います.　aとbの最初に学習率を入力してください．
結果の確認はcの学習済みニューロンのテストで行えます.
重みおよびバイアスはdを選択することでリセットできます.
終了したい場合は選択画面でescキーを押してください．

# Note

一括学習での損失の推移はloss_batch.csv, 逐次学習での損失の推移はloss_seq.csvに出力されます.
また，パラメータはw.csvに出力され，学習後のパラメータはw_batch.csv, w_seq.csvに出力されます．
