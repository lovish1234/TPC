python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-dot_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file certain_dot_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-dot_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file certain_dot_val

python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file certain_L2_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file certain_L2_val

python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-cosine_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file certain_cosine_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-cosine_distance-type_certain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file certain_cosine_val


python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-dot_distance-type_uncertain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_dot_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-dot_distance-type_uncertain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_dot_val

python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_uncertain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_uncertain_positive_vs_negative_same_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_val


python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_uncertain_positive_vs_negative_different_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_different_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-CE_distance-L2_distance-type_uncertain_positive_vs_negative_different_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_different_val

python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-MSE_distance-L2_distance-type_uncertain_positive_vs_negative_different_radius-type-linear_radius-which-pred/img/train/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_different_MSE_train
python tf2pd.py  ../uc-dpc/log_new/ucf11-128_r18_dpc-rnn_bs64_lr0.001_seq8_pred3_len5_ds3_train-all_loss-function-MSE_distance-L2_distance-type_uncertain_positive_vs_negative_different_radius-type-linear_radius-which-pred/img/val/  --write-csv --no-write-pkl --out-dir converted --out-file uncertain_L2_different_MSE_val
