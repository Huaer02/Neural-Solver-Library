# Neural-Solver-Library (NeuralSolver)

NeuralSolver is an open-source library for deep learning researchers, especially for neural PDE solvers.

## Features

This library current supports the following benchmarks:

- Standard Benchmarks

Here is the model list:

- [ ] **Transolver** - Transolver: A Fast Transformer Solver for PDEs on General Geometries [[ICML 2024]](https://arxiv.org/abs/2402.02366)

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
