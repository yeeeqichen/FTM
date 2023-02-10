# hyper params of FTM
TIME_LINE_LENGTH=1
N_DEGREE=10
N_LAYER=2
PATH_PREFIX="/path/to/data_fold/"
DATASET="reddit"

python3 run_FTM.py \
  --gpu 0 \
  --n_degree $N_DEGREE \
  --n_layer $N_LAYER \
  --prefix "TGN-FTM" \
  --data $DATASET \
  --use_memory \
  --use_time_line \
  --time_line_length $TIME_LINE_LENGTH \
  --sample_mode time \
  --hard_sample \
  --path_prefix $PATH_PREFIX \
  --bs 300