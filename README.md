# Enabling Lightweight Fine-tuning for Pre-trained Language Model Compression based on Matrix Product Operators

This is our Pytorch implementation for the paper:
> Peiyu Liu, Ze-Feng Gao, Wayne Xin Zhao, Z.Y. Xie, Zhong-Yi Lu and Ji-Rong Wen(2021). Enabling Lightweight Fine-tuning for Pre-trained Language Model Compression based on Matrix Product Operators
# Introduction
This paper presents a novel pre-trained language models (PLM) compression approach based on the matrix product operator (short as MPO) from quantum many-body physics. It can decompose an original matrix into central tensors (containing the core information) and auxiliary tensors (with only a small proportion of parameters). With the decomposed MPO structure, we propose a novel fine-tuning strategy by only updating the parameters from the auxiliary tensors, and design an optimization algorithm for MPO-based approximation over stacked network architectures. Our approach can be applied to the original or the compressed PLMs in a general way, which derives a lighter network and significantly reduces the parameters to be fine-tuned. Extensive experiments have demonstrated the effectiveness of the proposed approach in model compression, especially the reduction in fine-tuning parameters (91% reduction on average).

 ![image](images/fig-MPO.png)
 
For more details about the technique of MPOP, refer to our [paper](https://arxiv.org/abs/2106.02205), code is coming soon...
 # Release Notes
 First version: 2021/05/21

 # Installation
 ```shell
pip install -r requirements.txt
 ```
## Lightweight fine-tuning
In lightweight fine-tuning, we use original ALBERT without fine-tuning as to be compressed. By performing MPO decomposition on each weight matrix, we obtain four auxiliary tensors and one central tensor per tensor set. This provides a good initialization for the task-specific distillation. Refer to [run_all_albert_fine_tune.sh](https://github.com/lpyhdzx/MPOP/blob/ac958a78e1cf41d7f4117582a1aa2df3edf7e6fa/albert/run_all_albert_fine_tune.sh)

```shell
run_task 0 SST-2 500 2.7e-5 3.0 32 sst_lf 128 2.8e-6 word_embed,FFN_1,FFN_2,attention,pooler 480 384 256 albert-base-v2 -1 noload Noupdate --tensor_learn\ --pooler_trunc=256\ --load_best_model_at_end\ --metric_for_best_model="acc"\ --do_train
```
## Dimension squeezing
In Dimension squeezing, we compute approiate truncation order for the whole model. In order to re-produce the results in paper, we prepare the model after lightweight fine-tuning. Refer to [run_all_albert_fine_tune.sh](https://github.com/lpyhdzx/MPOP/blob/ac958a78e1cf41d7f4117582a1aa2df3edf7e6fa/albert/run_all_albert_fine_tune.sh)

albert models [google drive](https://drive.google.com/file/d/1shpcqfDemRaWhxIwcczDB_YePIyyF0bk/view?usp=sharing)

```shell
# e.g. SST-2
run_task 0 SST-2 500 2.7e-5 3.0 32 sst_rep 128 2.8e-6 word_embed 240 384 256 $check_point_dir/sst_paper1 -1 word_embed Noupdate --tensor_learn\ --pooler_trunc=256
```

## TODO
- [x] prepare data and code
- [x] upload models in order to reproduce experiments
- [ ] supplementary details for paper