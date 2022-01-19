mark=v2

data_path=../ECM/data/NLPCC2017_95W
train_data=${data_path}/train.txt

 

save_data=${data_path}/dict_${mark}

mkdir -p log
log=log/log_build_vocab_$mark
err=log/err_build_vocab_$mark

alias pythont='/home/wzw/anaconda3/envs/pytorch/bin/python3.7'
pythont build_vocab.py \
    -train_data $train_data \
    -save_data $save_data \
    -config ./config.yml 2> ${err} | tee ${log}
