set -x
base_dir=/home/liupeiyu/MPO_compression/sshlab
data_dir_base=/mnt/nlp_data/GLUE
check_point_dir=/mnt/liupeiyu/checkpoint/sshlab
gpu_num=$1
export WANDB_PROJECT=camera_ready

echo $gpu_num
function run_task() {
  export CUDA_VISIBLE_DEVICES=$1
  COMMON_ARGS="--data_dir="$data_dir_base/$2" --model_name_or_path=${14} --tokenizer_name=albert-base-v2 --evaluation_strategy=steps --eval_steps=100 --logging_steps=50 --overwrite_output_dir --save_steps=50000 --gpu_num=$1 --task_name=$2 --warmup_step=$3 --learning_rate=$4 --num_train_epochs=$5 --per_device_train_batch_size=$6 --output_dir="$check_point_dir/$7" --run_name=$7 --max_seq_length=$8 --mpo_lr=$9 --mpo_layers=${10} --emb_trunc=${11} --linear_trunc=${12} --attention_trunc=${13} --max_steps=${15} --load_layer=${16} --update_mpo_layer=${17} ${18}"
  nohup python $base_dir/run_glue_v5.py \
      ${COMMON_ARGS} \
      --do_predict \
      --do_eval > log_albert/$7.log 2>&1 &
}

############################ paper code
# sst
run_task 0 SST-2 500 2.7e-5 3.0 32 sst_rep 128 2.8e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sst_paper1 -1 word_embed Noupdate --tensor_learn\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# mnli
# run_task 3 MNLI 1000 3e-5 3.0 32 mnli_rep 128 2e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mnli_paper1/checkpoint-35500 -1 word_embed noupdate --pooler_trunc=1000\ --eval_steps=500\ --do_predict
# qnli
# run_task 3 QNLI 1986 1e-5 3.0 32 qnli_rep 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_paper3/checkpoint-6000 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --pooler_trunc=1000\ --eval_steps=500\ --do_predict\ --tensor_learn\ --save_steps=500
# cola
# run_task 1 CoLA 320 1e-5 1000.0 16 cola_rep 128 1e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 word_embed Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# sts-b
# run_task 7 STS-B 214 2e-5 1000.0 16 sts_rep 128 2e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 7000 word_embed  Noupdate --do_train\ --do_predict\ --pooler_trunc=1000\ --eval_steps=500\ --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="spearmanr"
# qqp
# run_task 2 QQP 1000 5e-5 1000.0 32 qqp_rep 128 5e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qqp_baseline 14000 word_embed Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --eval_steps=250\ --load_best_model_at_end\ --metric_for_best_model="acc"
# mrpc
# run_task 7 MRPC 200 2e-5 1000.0 32 mrpc_rep 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --do_train\ --pooler_trunc=1000\ --do_predict\ --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="acc"
# rte
# run_task 6 RTE 200 3e-5 1000.0 32 rte_rep 128 3e-5 word_embed 250 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/rte_albert_ori 800 word_embed Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn\ --do_predict
# wnli
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_rep 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 /mnt/liupeiyu/checkpoint/sshlab/wnli_baseline 800 word_embed,FFN_1,FFN_2,attention,pooler NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"