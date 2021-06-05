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
########## WNLI

######### SST-2
# 最优模型非balance
# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_baseline 128 2.8e-6 nompo 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_tensor_learn_v3 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_tensor_learn_v2 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --balance_attention

# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_novalue 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 409 274 149 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --balance_attention

# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_novalue_1 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 409 274 200 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 成功复现上面的结果说明代码没问题
# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_novalue_1_test 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 409 274 200 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 6 SST-2 500 2.7e-5 3.0 32 sst_paper1 128 2.8e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sst_baseline -1 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 6 SST-2 500 2.7e-5 3.0 32 sst_full_TL 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 3 SST-2 500 2.7e-5 3.0 32 sst_full 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=500

# run_task 6 SST-2 500 2.7e-5 3.0 32 sst_dir 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/sst_baseline -1 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=500\ --tensor_learn

######## QNLI
# run_task 2 QNLI 1986 9.98e-7 3.0 32 qnli_rankstep_allTL_lr 128 1.556e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 192 /mnt/checkpoint/sshlab/qnli_ori
# pooler
# run_task 2 QNLI 1986 9.98e-5 3.0 32 qnli_paper18_pooler 128 1.556e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 ${check_point_dir}/qnli_full_ori -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000
# 最优模型非balance
# run_task 0 QNLI 1986 9.98e-5 3.0 32 qnli_tensor_learn_v2 128 1.556e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 0 QNLI 1986 9.98e-5 3.0 32 qnli_tensor_learn_v3 128 1.556e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --balance_attention

# run_task 1 QNLI 1986 9.98e-5 3.0 32 qnli_novalue 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 362 223 131 /mnt/liupeiyu/checkpoint/sshlab/qnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --balance_attention

# run_task 1 QNLI 1986 9.98e-5 3.0 32 qnli_novalue_1 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 362 223 191 /mnt/liupeiyu/checkpoint/sshlab/qnli_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full


######## MNLI

# pooler版本
# 非balance的最优模型
# run_task 1 MNLI 1000 3e-5 3.0 32 mnli_tensor_learn_v3 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"

# run_task 1 MNLI 1000 3e-5 3.0 32 mnli_tensor_learn_v2 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --balance_attention

# run_task 2 MNLI 1000 3e-5 3.0 32 mnli_novalue 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 359 218 131 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --balance_attention

# run_task 2 MNLI 1000 3e-5 3.0 32 mnli_novalue_1 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 359 218 191 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 MNLI 1000 3e-5 3.0 32 mnli_novalue_2 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 359 218 131 albert-base-v2 -1 Noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 MNLI 1000 3e-5 3.0 32 mnli_novalue_3 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 245 250 albert-base-v2 -1 Noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000

# run_task 0 MNLI 1000 3e-5 3.0 32 mnli_baseline 128 2e-6 nompo 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --eval_steps=500

# run_task 1 MNLI 1000 3e-5 3.0 32 mnli_paper1 128 2e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mnli_baseline -1 noload noupdate --tensor_learn\ --do_train\ --do_predict\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --eval_steps=500
# run_task 3 MNLI 1000 3e-5 3.0 32 mnli_full 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --do_train\ --do_predict\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --eval_steps=500

# run_task 3 MNLI 1000 3e-5 3.0 32 mnli_full_TL 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --tensor_learn\ --do_train\ --do_predict\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --eval_steps=500

# run_task 0 MNLI 1000 3e-5 3.0 32 mnli_dir 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/mnli_baseline -1 noload noupdate --tensor_learn\ --do_train\ --do_predict\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="mnli/acc"\ --eval_steps=500

######## MRPC
# run_task 0 MRPC 200 2e-5 1000.0 32 mrpc_baseline 128 2e-5 nompo 1000 1000 1000 albert-base-v2 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# run_task 1 MRPC 200 2e-5 1000.0 32 mrpc_full 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# run_task 1 MRPC 200 2.96198e-05 3.0 8 mrpc_full_TL 128 2.96198e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn
# run_task 0 MRPC 200 2.96198e-05 3.0 8 mrpc_full_TL_2_test 128 2.96198e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_tensor_learn_hyper -1 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --tensor_learn\ --balance_attention

# run_task 1 MRPC 200 2e-5 1000.0 32 mrpc_dir 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn


# run_task 0 MRPC 200 2e-5 1000.0 32 mrpc_paper1 128 2e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_train\ --pooler_trunc=1000
# run_task 0 MRPC 200 2e-5 1000.0 32 mrpc_paper2 128 2e-5 word_embed 200 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_train\ --pooler_trunc=1000
# run_task 1 MRPC 200 2e-5 1000.0 32 mrpc_paper3 128 2e-5 word_embed 200 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --do_predict\ --tensor_learn
# run_task 7 MRPC 200 2e-5 1000.0 32 mrpc_paper4 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --do_predict\ --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="acc"
######## STS-B
# run_task 0 STS-B 214 2e-5 1000.0 16 sts_baseline 128 2e-5 nompo 1000 1000 1000 albert-base-v2 3598 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"
# run_task 0 STS-B 214 2e-5 1000.0 16 sts_full 128 2e-5 nompo 1000 1000 1000 albert-base-v2 3598 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"
# run_task 0 STS-B 214 2e-5 1000.0 16 sts_full_TL 128 2e-5 nompo 1000 1000 1000 albert-base-v2 3598 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"\ --tensor_learn
# run_task 0 STS-B 214 2e-5 1000.0 16 sts_dir 128 2e-5 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 3598 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"

# run_task 0 STS-B 214 2e-5 1000.0 16 sts_paper1 128 2e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 3598 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"

# run_task 0 STS-B 214 2e-5 1000.0 16 sts_paper2 128 2e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 7000 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="pearson"\ --eval_steps=500
# run_task 7 STS-B 214 2e-5 1000.0 16 sts_paper3 128 2e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 7000 Noload  Noupdate --do_train\ --do_predict\ --pooler_trunc=1000\ --eval_steps=500\ --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="spearmanr"

######## QQP

# run_task 1 QQP 0 4.18e-6 2000.0 32 qqp_TL_RS 128 1.1e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 192 /mnt/checkpoint/sshlab/qqp_tensor_learn_lr_2/ -1 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --tensor_learn\ --rank_step
# run_task 2 QQP 1000 5e-5 1000.0 32 qqp_tensor_learn_v2 128 5e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --balance_attention

# run_task 2 QQP 1000 5e-5 1000.0 32 qqp_tensor_learn_v3 128 5e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 1 QQP 1000 5e-5 1000.0 32 qqp_baseline 128 5e-5 nompo 1000 1000 1000 albert-base-v2 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=250

# run_task 0 QQP 1000 5e-6 1000.0 32 qqp_full 128 5e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 14000 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=250

# run_task 1 QQP 1000 5e-6 1000.0 32 qqp_full_TL 128 5e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=250

# run_task 1 QQP 1000 5e-6 1000.0 32 qqp_dir_142 128 5e-6 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/qqp_baseline 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=500


# run_task 2 QQP 1000 5e-5 1000.0 32 qqp_paper1 128 5e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qqp_baseline 14000 Noload Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --eval_steps=250\ --load_best_model_at_end\ --metric_for_best_model="acc"


######## cola
# run_task 2 CoLA 320 1e-5 1000.0 16 cola_tensor_learn_v2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train
# run_task 0 CoLA 320 1e-5 1000.0 16 cola_tensor_learn_balance 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --balance_attention\ --do_train

# ----压缩
# run_task 2 CoLA 0 1e-5 50.0 16 cola_TL_RS_paper18_pooler_20 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 409 273 147 /mnt/liupeiyu/checkpoint/sshlab/cola_albert_ori -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --balance_attention

# run_task 1 CoLA 0 1e-5 50.0 16 cola_novalue_1 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 304 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 CoLA 0 1e-5 50.0 16 cola_novalue_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 409 273 147 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_balance -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --balance_attention
# 使用ori
# run_task 2 CoLA 0 1e-5 50.0 16 cola_novalue_3 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 304 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

############### 用train/dev 按照0.8区分后结果
# run_task 2 SST-2 500 2.7e-5 3.0 32 sst_tr_dev_ori 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 192 albert-base-v2 -1 Noload Noupdate --do_train\ --do_predict
# run_task 2 SST-2 500 2.7e-5 3.0 32 sst_tr_dev_ori 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 192 albert-base-v2 -1 Noload Noupdate --do_train\ --do_predict\ --do_split\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --do_predict

############################################ camera ready
# run_task 1 CoLA 0 1e-5 50.0 16 cola_TL_RS_paper18_pooler_19 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 304 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 304 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_3 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 250 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 CoLA 0 1e-5 50.0 16 cola_paper_4 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 240 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 CoLA 0 1e-5 50.0 16 cola_paper_5 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 220 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_6 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_7 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 180 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_8 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 280 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

########## 这里加入了step_train 参数，默认为False就是不用mask

# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_8_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 280 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 CoLA 0 1e-5 50.0 16 cola_paper_9 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_4_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 240 384 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_10 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 这个日志已经被覆盖无法查看，但是从wandb中看到的效果不好
# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_11 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 380 256 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 后续还是从attention入手
# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_12 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 240 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 重新修改了dense，value都截断
# run_task 1 CoLA 0 1e-5 50.0 16 cola_paper_10_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 CoLA 0 1e-5 50.0 16 cola_paper_12_2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 240 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 重新修改了dense截断，value不截断
# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_10_3 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 250 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 2 CoLA 0 1e-5 50.0 16 cola_paper_12_3 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 265 384 240 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

#cola结论：不要截断attention中的dense，


# 这两个都不好
# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_u1 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 409 273 147 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full
# run_task 0 CoLA 0 1e-5 50.0 16 cola_paper_1 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 409 273 147 /mnt/liupeiyu/checkpoint/sshlab/cola_tensor_learn_balance -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --balance_attention

# run_task 1 CoLA 320 1e-5 1000.0 16 cola_baseline 128 1e-5 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500

# run_task 0 COLA 320 4.08046e-05 1000.0 16 cola_baseline_2 128 4.08046e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 0 COLA 320 2e-05 1000.0 16 cola_baseline_3 128 2e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 0 COLA 320 1.8e-05 1000.0 16 cola_baseline_4 128 1.8e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 2 COLA 320 2.5e-05 1000.0 16 cola_baseline_5 128 2e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 2 COLA 320 3e-05 1000.0 16 cola_baseline_6 128 3e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 2 COLA 320 3.5e-05 1000.0 16 cola_baseline_7 128 3.5e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500
# run_task 2 COLA 320 3.2e-05 1000.0 16 cola_baseline_8 128 3.2e-05 nompo 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500


# run_task 0 COLA 0 4.08046e-05 1000.0 16 cola_full 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500\ --weight_decay=0.0975991
# run_task 2 COLA 0 4.08046e-05 1000.0 16 cola_full2 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991
# run_task 2 COLA 0 4.08046e-05 1000.0 16 cola_full3 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991
# run_task 2 COLA 320 1e-06 1000.0 16 cola_full4 128 1e-06 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991

# run_task 2 COLA 0 4.08046e-05 1000.0 16 cola_dir 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991

# run_task 0 COLA 320 1e-5 1000.0 16 cola_full_TL 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500

# run_task 0 COLA 0 4.08046e-05 1000.0 16 cola_dir 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500\ --weight_decay=0.0975991

# run_task 1 CoLA 320 1e-5 1000.0 16 cola_paper1 128 1e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500

# run_task 1 CoLA 320 1e-5 1000.0 16 cola_paper1 128 1e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=500

# run_task 7 CoLA 320 1e-5 1000.0 16 cola_paper2 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=250

# run_task 6 CoLA 320 1e-5 50.0 16 cola_paper3 128 1e-5 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline -1 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=250

# run_task 2 COLA 0 4.08046e-05 1000.0 16 cola_paper4 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991
# run_task 0 COLA 0 4.08046e-05 1000.0 16 cola_paper4_test 128 1.0955e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991

# run_task 1 COLA 0 1e-05 1000.0 16 cola_paper5 128 1e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100
# run_task 2 COLA 0 1e-05 1000.0 16 cola_paper6 128 1e-05 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100
# run_task 2 COLA 320 1e-05 1000.0 16 cola_paper7 128 1e-05 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100
# run_task 0 COLA 320 1e-05 1000.0 16 cola_paper8 128 1e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100

# run_task 1 COLA 320 1.8e-05 1000.0 16 cola_paper9 128 1.8e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100\ --weight_decay=0.0975991
# run_task 1 COLA 320 1.8e-05 1000.0 16 cola_paper10 128 1.8e-05 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/cola_baseline 5336 Noload Noupdate --tensor_learn\ --load_best_model_at_end\ --metric_for_best_model="mcc"\ --do_train\ --do_predict\ --eval_steps=100



# --------- sst-2
# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_paper_1 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 0 SST-2 500 2.7e-5 3.0 32 sst_paper_2 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 384 256 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 SST-2 500 2.7e-5 3.0 32 sst_paper_3 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 384 250 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# 结论是看样子稍微减少一点MLP点维度也是可以提升效果的
# run_task 1 SST-2 500 2.7e-5 3.0 32 sst_paper_4 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 380 256 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 SST-2 500 2.7e-5 3.0 32 sst_paper_5 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 370 256 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 SST-2 500 2.7e-5 3.0 32 sst_paper_6 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 360 256 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full

# run_task 1 SST-2 500 2.7e-5 3.0 32 sst_paper_7 128 2.8e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sst_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full


# --------- QNLI
# run_task 2 QNLI 1986 9.98e-5 3.0 32 qnli_baseline 128 2.8e-6 nompo 1000 1000 1000 albert-base-v2 -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 0 QNLI 1986 1e-5 3.0 32 qnli_baseline 128 2.8e-6 nompo 1000 1000 1000 albert-base-v2 -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 0 QNLI 1986 1e-5 3.0 32 qnli_full 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 1 QNLI 1986 1e-5 3.0 32 qnli_full_TL 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn

# run_task 2 QNLI 1986 1e-5 3.0 32 qnli_dir 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/qnli_baseline -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn


# run_task 1 QNLI 1986 9.98e-5 3.0 32 qnli_paper_1 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 /mnt/liupeiyu/checkpoint/sshlab/qnli_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=200

# run_task 1 QNLI 1986 9.98e-5 3.0 32 qnli_paper_2 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 384 256 /mnt/liupeiyu/checkpoint/sshlab/qnli_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=200

# run_task 0 QNLI 1986 9.98e-5 3.0 32 qnli_paper_3 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 370 256 /mnt/liupeiyu/checkpoint/sshlab/qnli_tensor_learn_v2 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=150

# run_task 1 QNLI 1986 1e-5 3.0 32 qnli_paper1 128 2.8e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_baseline -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --do_predict

# run_task 1 QNLI 1986 1e-5 3.0 32 qnli_paper2 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_baseline -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --do_predict\ --tensor_learn

# run_task 3 QNLI 1986 1e-5 3.0 32 qnli_paper3 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_baseline -1 noload noupdate --do_train\ --pooler_trunc=1000\ --eval_steps=500\ --do_predict\ --tensor_learn\ --save_steps=500



# --------- MNLI
# run_task 2 MNLI 1000 3e-5 3.0 32 mnli_paper_1 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=500

# run_task 2 MNLI 1000 3e-5 3.0 32 mnli_paper_2 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 384 256 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=500

# run_task 0 MNLI 1000 3e-5 3.0 32 mnli_paper_3 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 250 370 256 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=150

# run_task 1 MNLI 1000 3e-5 3.0 32 mnli_paper_4 128 2e-6 word_embed,FFN_1,FFN_2,attention,pooler 265 384 250 /mnt/liupeiyu/checkpoint/sshlab/mnli_tensor_learn_v3 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=150
# --------- QQP
# run_task 6 QQP 1000 5e-5 1000.0 32 qqp_paper_1 128 5e-5 word_embed,FFN_1,FFN_2,attention,pooler 200 384 256 /mnt/liupeiyu/checkpoint/sshlab/qqp_tensor_learn_v3 14000 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=250

# run_task 6 QQP 1000 5e-5 1000.0 32 qqp_paper_2 128 5e-5 word_embed,FFN_1,FFN_2,attention,pooler 250 384 256 /mnt/liupeiyu/checkpoint/sshlab/qqp_tensor_learn_v3 14000 word_embed,FFN_1,FFN_2,attention,pooler Noupdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_full\ --eval_steps=250

# --------- MRPC
# run_task 0 MRPC 200 2.96198e-05 3.0 8 mrpc_tensor_learn_v2 128 2.96198e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# --------- STS-B
# run_task 0 STS-B 214 2.0e-05 1000.0 16 sts_tensor_learn_v2 128 2.0e-05 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 3598 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="spearmanr"

# --------- RTE
# run_task 1 RTE 200 3e-5 1000.0 32 rte_albert_ori 128 3e-5 Nompolayer 1000 1000 1000 albert-base-v2 800 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# run_task 2 RTE 200 3e-5 1000.0 32 rte_dir 128 3e-5 word_embed,FFN_1,FFN_2,attention,pooler 385 308 208 /mnt/liupeiyu/checkpoint/sshlab/rte_albert_ori 800 Noload  Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn



# run_task 1 RTE 200 3e-5 1000.0 32 rte_paper1 128 3e-5 word_embed 250 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/rte_albert_ori 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# run_task 6 RTE 200 3e-5 1000.0 32 rte_paper2 128 3e-5 word_embed 250 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/rte_albert_ori 800 Noload Noupdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --tensor_learn\ --do_predict

# --------- WNLI
# run_task 0 WNLI 200 1.1e-04 3.0 8 wnli_tensor_learn_v2 128 3.86e-5 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 -1 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --weight_decay=0.290875

# run_task 0 WNLI 200 2e-5 3.0 8 wnli_baseline 128 -1 nompo 1000 1000 1000 albert-base-v2 -1 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_full 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 1000 1000 1000 albert-base-v2 800 noload NoUpdate --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --eval_steps=100

# run_task 1 WNLI 200 2e-5 3.0 8 wnli_dir 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 82 72 45 /mnt/liupeiyu/checkpoint/sshlab/wnli_baseline -1 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# 测试集合
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_baseline_test 128 -1 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/wnli_baseline -1 noload NoUpdate --tensor_learn\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --do_predict


# run_task 0 WNLI 200 2e-5 3.0 8 wnli_paper1 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 /mnt/liupeiyu/checkpoint/sshlab/wnli_baseline 800 noload NoUpdate --tensor_learn\ --do_train\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_paper1_test 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 /mnt/liupeiyu/checkpoint/sshlab/wnli_paper1 800 noload NoUpdate --tensor_learn\ --do_predict\ --pooler_trunc=1000 
# run_task 0 WNLI 200 2e-5 3.0 8 wnli_paper1_test2 128 -1 word_embed,FFN_1,FFN_2,attention,pooler 75 82 45 /mnt/liupeiyu/checkpoint/sshlab/wnli_paper1 800 word_embed,FFN_1,FFN_2,attention,pooler NoUpdate --tensor_learn\ --do_predict\ --pooler_trunc=1000 

################ make all predict result MPO
# ----- SST-2
# sst_paper1
# ----- MNLI
# run_task 3 MNLI 1000 3e-5 3.0 32 mnli_paper1_predict 128 2e-6 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mnli_paper1/checkpoint-35500 -1 word_embed noupdate --pooler_trunc=1000\ --eval_steps=500\ --do_predict

# ----- QNLI
# run_task 3 QNLI 1986 1e-5 3.0 32 qnli_paper3_p 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_paper3/checkpoint-6000 -1 word_embed,FFN_1,FFN_2,attention,pooler noupdate --pooler_trunc=1000\ --eval_steps=500\ --do_predict\ --tensor_learn\ --save_steps=500

# ----- CoLA
# cola_paper1

# ----- STS-B
# sts_paper3

# ----- QQP
# run_task 7 QQP 1000 5e-5 1000.0 32 qqp_paper1_predict 128 5e-5 word_embed 240 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qqp_paper1 14000 word_embed Noupdate --tensor_learn\ --pooler_trunc=1000\ --eval_steps=250\ --load_best_model_at_end\ --metric_for_best_model="acc"
# 使用qqp_paper1

# ----- MRPC
# mrpc_paper4

# ----- RTE
# 这个已经废弃
# run_task 2 RTE 200 3e-5 1000.0 32 rte_paper1_p 128 3e-5 word_embed 250 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/rte_paper1 800 word_embed Noupdate --do_predict\ --pooler_trunc=1000
# 使用rte_paper2
# rte_paper2

# ----- WNLI
# 已完成
################ make all predict result albert baseline
# ----- SST-2
# run_task 6 SST-2 500 2.7e-5 3.0 32 sst_baseline_p 128 2.8e-6 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sst_baseline -1 Noload Noupdate --tensor_learn\ --do_predict

# ----- MNLI
# run_task 6 MNLI 1000 3e-5 3.0 32 mnli_baseline_p 128 2e-6 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mnli_baseline -1 noload noupdate --pooler_trunc=1000\ --eval_steps=500\ --do_predict

# ----- QNLI
# run_task 6 QNLI 1986 1e-5 3.0 32 qnli_baseline_p 128 2.8e-6 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qnli_baseline -1 noload noupdate --do_predict\ --pooler_trunc=1000

# ----- CoLA
# cola_baseline
# ----- STS-B
# run_task 6 STS-B 214 2e-5 1000.0 16 sts_baseline_p 128 2e-5 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/sts_baseline 3598 Noload  Noupdate --pooler_trunc=1000\ --do_predict

# ----- QQP
# run_task 6 QQP 1000 5e-5 1000.0 32 qqp_baseline_p 128 5e-5 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/qqp_baseline 14000 Noload Noupdate --tensor_learn\ --do_predict\ --pooler_trunc=1000\ 

# ----- MRPC
# run_task 7 MRPC 200 2e-5 1000.0 32 mrpc_baseline_p 128 2e-5 nompo 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/mrpc_baseline 800 Noload Noupdate --do_predict\ --pooler_trunc=1000

# ----- RTE
# run_task 6 RTE 200 3e-5 1000.0 32 rte_baseline_p 128 3e-5 Nompolayer 1000 1000 1000 /mnt/liupeiyu/checkpoint/sshlab/rte_albert_ori 800 Noload  Noupdate --do_predict\ --pooler_trunc=1000\ --load_best_model_at_end\ --metric_for_best_model="acc"

# ----- WNLI
# 已经完成