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
  nohup python run_glue_v5.py \
      ${COMMON_ARGS} \
      --do_eval > log_albert/$7.log 2>&1 &
}

############################ lightweight fine-tuning
# sst-2
# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_lf 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --do_train
# mnli
# run_task 6 MNLI 1000 3e-5 3.0 32 mnli_rep 128 2e-6 word_embed 240 384 256 albert-base-v2 -1 noload noupdate --pooler_trunc=256\ --eval_steps=500\ --do_train
# qnli
# run_task 6 QNLI 1986 1e-5 3.0 32 qnli_rep 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 384 256 albert-base-v2 -1 noload noupdate --pooler_trunc=256\ --eval_steps=500\ --tensor_learn\ --save_steps=500\ --do_train
# cola
# run_task 6 CoLA 320 1e-5 1000.0 16 cola_rep 128 1e-5 word_embed 240 384 256 albert-base-v2 5336 noload Noupdate --tensor_learn\ --eval_steps=500\ --pooler_trunc=256\ --do_train
# sts-b
# run_task 6 STS-B 214 2e-5 1000.0 16 sts_rep 128 2e-5 word_embed 240 384 256 albert-base-v2 7000 noload  Noupdate --pooler_trunc=256\ --eval_steps=500\ --tensor_learn\ --do_train
# qqp
# run_task 0 QQP 1000 5e-5 1000.0 32 qqp_rep 128 5e-5 word_embed 240 384 256 albert-base-v2 14000 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --eval_steps=250\ --do_train
# mrpc
# run_task 0 MRPC 200 2e-5 1000.0 32 mrpc_rep 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 albert-base-v2 800 noload Noupdate --pooler_trunc=256\ --tensor_learn\ --do_train
# rte
# run_task 6 RTE 200 3e-5 1000.0 32 rte_rep 128 3e-5 word_embed 250 384 256 albert-base-v2 800 noload Noupdate --pooler_trunc=256\ --tensor_learn\ --do_train
# wnli
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_rep 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 albert-base-v2 800 noload NoUpdate --tensor_learn\ --pooler_trunc=256\ --do_train
############################ dimension squeezing
# sst-2
# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_rep 128 2.8e-6 word_embed 240 384 256 $check_point_dir/sst_paper1 -1 word_embed Noupdate --tensor_learn\ --pooler_trunc=256
# mnli
# run_task 6 MNLI 1000 3e-5 3.0 32 mnli_rep 128 2e-6 word_embed 240 384 256 $check_point_dir/mnli_paper1 -1 word_embed noupdate --pooler_trunc=256\ --eval_steps=500
# qnli
# run_task 6 QNLI 1986 1e-5 3.0 32 qnli_rep 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 384 256 $check_point_dir/qnli_paper3/checkpoint-6000 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --pooler_trunc=256\ --eval_steps=500\ --tensor_learn\ --save_steps=500
# cola
# run_task 6 CoLA 320 1e-5 1000.0 16 cola_rep 128 1e-5 word_embed 240 384 256 $check_point_dir/cola_paper9 5336 word_embed Noupdate --tensor_learn\ --eval_steps=500\ --pooler_trunc=256
# sts-b
# run_task 6 STS-B 214 2e-5 1000.0 16 sts_rep 128 2e-5 word_embed 240 384 256 $check_point_dir/sts_paper3 7000 word_embed  Noupdate --pooler_trunc=256\ --eval_steps=500\ --tensor_learn
# qqp
# run_task 0 QQP 1000 5e-5 1000.0 32 qqp_rep 128 5e-5 word_embed 240 384 256 $check_point_dir/qqp_paper1 14000 word_embed Noupdate --tensor_learn\ --pooler_trunc=256\ --eval_steps=250
# mrpc
# run_task 0 MRPC 200 2e-5 1000.0 32 mrpc_rep 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 $check_point_dir/mrpc_paper4 800 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --pooler_trunc=256\ --tensor_learn
# rte
# run_task 6 RTE 200 3e-5 1000.0 32 rte_rep 128 3e-5 word_embed 250 384 256 $check_point_dir/rte_paper1 800 word_embed Noupdate --pooler_trunc=256\ --tensor_learn
# wnli
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_rep 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 $check_point_dir/wnli_paper1 800 word_embed,FFN_1,FFN_2,attention,pooler NoUpdate --tensor_learn\ --pooler_trunc=256