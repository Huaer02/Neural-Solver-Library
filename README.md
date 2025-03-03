# Neural-Solver-Library (NeuralSolver)

NeuralSolver is an open-source library for deep learning researchers, especially for neural PDE solvers.

## Features

This library current supports the following benchmarks:

- Standard Benchmarks

Here is the model list:

- [x] **Transolver** - Transolver: A Fast Transformer Solver for PDEs on General Geometries [[ICML 2024]](https://arxiv.org/abs/2402.02366)
- [x] **Factformer** - Scalable Transformer for PDE Surrogate Modeling [[NeurIPS 2023]](https://arxiv.org/abs/2305.17560)
- [x] **LSM** - Solving High-Dimensional PDEs with Latent Spectral Models [[ICML 2023]](https://arxiv.org/pdf/2301.12664)
- [x] **GNOT** - GNOT: A General Neural Operator Transformer for Operator Learning [[ICML 2023]](https://arxiv.org/abs/2302.14376)
- [x] **Galerkin Transformer** - Choose a Transformer: Fourier or Galerkin [[NeurIPS 2021]](https://arxiv.org/abs/2105.14995)
- [x] **FNO** - Fourier Neural Operator for Parametric Partial Differential Equations [[ICLR 2021]](https://arxiv.org/pdf/2010.08895)
- [x] **Transformer** - Attention Is All You Need [[NeurIPS 2017]](https://arxiv.org/pdf/1706.03762)

Some vision backbones can be good baselines for tasks in structured geometries.

- [x] **Swin Transformer** - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows [[ICCV 2021]](https://arxiv.org/abs/2103.14030)
- [x] **U-Net** - U-Net: Convolutional Networks for Biomedical Image Segmentation [[MICCAI 2015]](https://arxiv.org/pdf/1505.04597)


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data
3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
bash ./scripts/Transolver_pipe.sh
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transolver.py`.
- Include the newly added model in the `model_dict` of `./models/model_factory.py`.
- Create the corresponding scripts under the folder `./scripts`.

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2024Transolver,
  title={Transolver: A Fast Transformer Solver for PDEs on General Geometries},
  author={Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## Contact

If you have any questions or want to use the code, please contact [wuhx23@mails.tsinghua.edu.cn](mailto:wuhx23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Transolver

https://github.com/thuml/Latent-Spectral-Models

https://github.com/neuraloperator/neuraloperator

https://github.com/neuraloperator/Geo-FNO
