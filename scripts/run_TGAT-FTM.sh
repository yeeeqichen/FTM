# hyper params of FTM
TIME_LINE_LENGTH=1
N_DEGREE=10
N_LAYER=2
PATH_PREFIX="/path/to/data_fold/"
DATASET="wikipedia"

python3 run_FTM.py \
  --use_time_line \
  --gpu 0 \
  --time_line_length $TIME_LINE_LENGTH \
  --sample_mode time \
  --hard_sample \
  --n_degree $N_DEGREE \
  --n_layer $N_LAYER \
  --bs 100 \
  --prefix "TGAT-FTM" \
  --path_prefix $PATH_PREFIX \
  --data $DATASET