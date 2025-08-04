# OrthoSolver for 1D datasets

# 13.03G
# Advection数据集
python run.py \
--gpu 0 \
--data_path ./data/PDEBench/1D/Advection/1D_Advection_Sols_beta0.1.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler OneCycleLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model OrthoSolver \
--n_hidden 64 \
--n_heads 8 \
--n_layers 2 \
--slice_num 64 \
--batch-size 16 \
--epochs 500 \
--eval 0 \
--loss_type l2 \
--use_multitask True \
--use_dwa True \
--lr 0.001 \
--save_name pdebench_1d_advection_beta1e0_1024_10_10_OrthoSolver

# Burgers数据集
python run.py \
--gpu 0 \
--data_path ./data/PDEBench_merge/1D/Burgers/1D_Burgers_Sols_Nu0.001.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler OneCycleLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model OrthoSolver \
--n_hidden 64 \
--n_heads 8 \
--n_layers 2 \
--slice_num 64 \
--batch-size 96 \
--epochs 500 \
--eval 0 \
--loss_type l2 \
--use_multitask True \
--use_dwa True \
--lr 0.001 \
--save_name pdebench_1d_burgers_nu0001_1024_10_10_OrthoSolver

# CFD数据集
python run.py \
--gpu 0 \
--data_path ./data/PDEBench_merge/1D/CFD/1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler OneCycleLR \
--space_dim 1 \
--fun_dim 30 \
--out_dim 3 \
--T_in 10 \
--T_out 10 \
--model OrthoSolver \
--n_hidden 64 \
--n_heads 8 \
--n_layers 2 \
--slice_num 64 \
--batch-size 96 \
--epochs 500 \
--eval 0 \
--loss_type l2 \
--use_multitask True \
--use_dwa True \
--lr 0.001 \
--save_name pdebench_1d_cfd_eta001_zeta001_1024_10_10_OrthoSolver

# ReactionDiffusion数据集
python run.py \
--gpu 0 \
--data_path ./data/PDEBench_merge/1D/ReactionDiffusion/ReacDiff_Nu0.5_Rho1.0.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler OneCycleLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model OrthoSolver \
--n_hidden 64 \
--n_heads 8 \
--n_layers 2 \
--slice_num 64 \
--batch-size 96 \
--epochs 500 \
--eval 0 \
--loss_type l2 \
--use_multitask True \
--use_dwa True \
--lr 0.001 \
--save_name pdebench_1d_reacdiff_nu05_rho10_1024_10_10_OrthoSolver

# diffusion-sorption数据集
python run.py \
--gpu 0 \
--data_path ./data/PDEBench_merge/1D/diffusion-sorption/1D_diff-sorp_NA_NA.hdf5 \
--loader pdebench_unified \
--geotype structured_1D \
--task dynamic_autoregressive \
--scheduler OneCycleLR \
--space_dim 1 \
--fun_dim 10 \
--out_dim 1 \
--T_in 10 \
--T_out 10 \
--model OrthoSolver \
--n_hidden 64 \
--n_heads 8 \
--n_layers 2 \
--slice_num 64 \
--batch-size 96 \
--epochs 200 \
--eval 0 \
--loss_type l2 \
--use_multitask True \
--use_dwa True \
--lr 0.001 \
--save_name pdebench_1d_diffsorp_na_na_1024_10_10_OrthoSolver
