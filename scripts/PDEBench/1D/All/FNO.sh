# 17.7G

python run.py \
--gpu 1 \
--data_path ./data/PDEBench_merge/1D/Advection/1D_Advection_Sols_beta0.1.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler StepLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--batch-size 256 \
--epochs 500 \
--eval 0 \
--save_name pdebench_1d_advection_beta1e0_1024_10_10_FNO

# Burgers数据集
python run.py \
--gpu 1 \
--data_path ./data/PDEBench_merge/1D/Burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler StepLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--batch-size 256 \
--epochs 500 \
--eval 0 \
--save_name pdebench_1d_burgers_nu0001_1024_10_10_FNO

# CFD数据集
python run.py \
--gpu 1 \
--data_path ./data/PDEBench_merge/1D/CFD/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler StepLR \
--space_dim 1 \
--fun_dim 30 \
--out_dim 3 \
--T_in 10 \
--T_out 10 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--batch-size 256 \
--epochs 500 \
--eval 0 \
--save_name pdebench_1d_cfd_eta001_zeta001_1024_10_10_FNO

# ReactionDiffusion数据集
python run.py \
--gpu 1 \
--data_path ./data/PDEBench_merge/1D/ReactionDiffusion/ReacDiff_Nu0.5_Rho1.0.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler StepLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--batch-size 256 \
--epochs 500 \
--eval 0 \
--save_name pdebench_1d_reacdiff_nu05_rho10_1024_10_10_FNO

# diffusion-sorption数据集
python run.py \
--gpu 1 \
--data_path ./data/PDEBench_merge/1D/diffusion-sorption/1D_diff-sorp_NA_NA.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler StepLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model FNO \
--n_hidden 64 \
--n_heads 8 \
--n_layers 8 \
--batch-size 256 \
--epochs 500 \
--eval 0 \
--save_name pdebench_1d_diffsorp_na_na_1024_10_10_FNO
