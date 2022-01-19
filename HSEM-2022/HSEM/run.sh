vocab_mark=v2

mark=$1
GPUID=$2


data_path=../ECM/data/NLPCC2017_95W

#data_path=../ECM/data/NLPCC2017

#data_path=../ECM/data/ESTC

train_data=${data_path}/train.txt
vocab_file=${data_path}/dict_${vocab_mark}.vocab.pt

config_from_local_or_loaded_model=0 # 0:local(from ./config.yml) 1:loaded model(from output_model/${modelMark}/config.yml)

mkdir -p log
log=log/log_$mark
err=log/err_$mark
model_output_dir=output_models/${mark}

python train.py \
    -gpuid ${GPUID} \
    -config ./config.yml \
    -config_with_loaded_model ./config.yml \
    -config_from_local_or_loaded_model ${config_from_local_or_loaded_model} \
    -train_data $train_data \
    -out_dir $model_output_dir \
    -vocab ${vocab_file} 2> ${err} | tee ${log}
