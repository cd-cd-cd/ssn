# python train_fiq.py \
#   --dataset 'fashion_iq' \
#   --model 'ssn_raw copy' \
#   --projection_dim 512 \
#   --hidden_dim 512 \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 5e-5 \
#   --lr_co 6.5e-5 \
#   --lr_sa 5e-5 \
#   --lr_ratio 0.2 \
#   --lr_gamma 0.1 \
#   --lr_step_size 10 \
#   --save_training \
#   --save_best \
#   --validation_frequency 1 \
#   --kl_weight 1 \
#   --n_layers 4 \
#   --n_heads 8 \
#   --project_name 'ssn' \
#   --workspace 'cd-cd-cd' \
#   --api_key 'Y5hPtJ89eMK2DDNdLOnFCdIxE' \


python train_fiq.py \
  --dataset 'fashion_iq' \
  --model 'ssn_crossAttention4_fusion3' \
  --projection_dim 512 \
  --hidden_dim 512 \
  --num_epochs 50 \
  --batch_size 128 \
  --lr 5e-5 \
  --lr_ratio 0.2 \
  --lr_gamma 0.1 \
  --lr_step_size 10 \
  --save_training \
  --save_best \
  --validation_frequency 1 \
  --kl_weight 1 \
  --n_layers 4 \
  --n_heads 8 \


# python train_cirr.py \
#   --dataset 'cirr' \
#   --model 'ssn_crossAttention4' \
#   --projection_dim 512 \
#   --hidden_dim 512 \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 5e-5 \
#   --lr_ratio 0.2 \
#   --lr_gamma 0.1 \
#   --lr_step_size 10 \
#   --save_training \
#   --save_best \
#   --validation_frequency 1 \
#   --kl_weight 1 \
#   --n_layers 4 \
#   --n_heads 8 \


# python train_fiq_copy.py \
#   --dataset 'cirr' \
#   --dataset_list_path '/amax/home/chendian/DQU-CIR-main/cirr_but_dataset_list.pkl' \
#   --model 'ssn_crossAttention4' \
#   --projection_dim 512 \
#   --hidden_dim 512 \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 5e-5 \
#   --lr_ratio 0.2 \
#   --lr_gamma 0.1 \
#   --lr_step_size 10 \
#   --save_training \
#   --save_best \
#   --validation_frequency 1 \
#   --kl_weight 1 \
#   --n_layers 4 \
#   --n_heads 8 \


# python test_dqu.py \
#   --dataset 'cirr' \
#   --model 'ssn_crossAttention4' \
#   --projection_dim 512 \
#   --hidden_dim 512 \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 5e-5 \
#   --lr_ratio 0.2 \
#   --lr_gamma 0.1 \
#   --lr_step_size 10 \
#   --save_training \
#   --save_best \
#   --validation_frequency 1 \
#   --kl_weight 1 \
#   --n_layers 4 \
#   --n_heads 8 \

# python -m debugpy --listen 1111 --wait-for-client train_fiq.py \
#   --dataset 'fashion_iq' \
#   --model 'ssn_crossAttention4' \
#   --projection_dim 512 \
#   --hidden_dim 512 \
#   --num_epochs 50 \
#   --batch_size 128 \
#   --lr 5e-5 \
#   --lr_ratio 0.2 \
#   --lr_gamma 0.1 \
#   --lr_step_size 10 \
#   --save_training \
#   --save_best \
#   --validation_frequency 1 \
#   --kl_weight 1 \
#   --n_layers 4 \
#   --n_heads 8 \
#   --lr_co 6e-05 \
#   --lr_sa 6e-05 \