# OrthoSolver

# diff-react
python run.py \
    --gpu 1 \
    --data_path ./data/PDEBench/2D/diff-react/2D_diff-react_NA_NA.hdf5 \
    --loader pdebench_unified \
    --geotype structured_2D \
    --task dynamic_autoregressive \
    --scheduler StepLR \
    --space_dim 2 \
    --downsamplex 2 \
    --downsampley 2 \
    --fun_dim 20 \
    --out_dim 2 \
    --T_in 10 \
    --T_out 10 \
    --model OrthoSolver \
    --n_hidden 64 \
    --n_heads 8 \
    --n_layers 2 \
    --slice_num 64 \
    --batch-size 1 \
    --epochs 200 \
    --eval 0 \
    --loss_type l2 \
    --use_multitask True \
    --use_dwa True \
    --lr 0.001 \
    --save_name pdebench_2d_diff_react_64_10_10_OrthoSolver

# CFD
# 24.97G
python run.py \
    --gpu 1 \
    --data_path ./data/PDEBench_merge/2D/CFD/2D_Train_Rand/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
    --loader pdebench_unified \
    --geotype structured_2D \
    --task dynamic_autoregressive \
    --scheduler StepLR \
    --space_dim 2 \
    --downsamplex 2 \
    --downsampley 2 \
    --fun_dim 40 \
    --out_dim 4 \
    --T_in 10 \
    --T_out 10 \
    --model OrthoSolver \
    --n_hidden 64 \
    --n_heads 8 \
    --n_layers 2 \
    --slice_num 64 \
    --batch-size 8 \
    --epochs 200 \
    --eval 0 \
    --loss_type l2 \
    --use_multitask True \
    --use_dwa True \
    --lr  0.001 \
    --save_name pdebench_2d_CFD_64_10_10_OrthoSolver