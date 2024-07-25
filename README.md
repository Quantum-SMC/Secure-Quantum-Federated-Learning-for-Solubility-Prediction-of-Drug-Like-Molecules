# Secure-Quantum-Federated-Learning-for-Solubility-Prediction-of-Drug-Like-Molecules
This project enables several pharmaceitical companies to jointly perform federated learning for training a GCN model that predict solubility of drug-like molecules.

For more detail refer to [Paper]()

The project is based on code by the authors of [SAFEFL](https://ieeexplore.ieee.org/abstract/document/10188630?casa_token=FDcsggxqcvwAAAAA:_JzLcQrYYbLfTwa_KSVOoy8iiDttXEeQ1y33HhvqEJl0BdmfaYHBXVS44Hx5IbdbRKNJTla8dg) and follows their general structure. 
The original code is available [here](https://github.com/encryptogroup/SAFEFL?tab=readme-ov-file).

## Aggregation rules
The following aggregation rules are added:

- [FedAvg](https://arxiv.org/abs/1602.05629)
- [Krum](https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html)
- [Trimmed mean](https://arxiv.org/abs/1803.01498)
- [Median](https://arxiv.org/abs/1803.01498)
- [FLTrust](https://arxiv.org/abs/2012.13995)
- [FLAME](https://arxiv.org/abs/2101.02281)
- [FLOD](https://eprint.iacr.org/2021/993)
- [ShieldFL](https://ieeexplore.ieee.org/document/9762272)
- [DnC](https://www.ndss-symposium.org/ndss-paper/manipulating-the-byzantine-optimizing-model-poisoning-attacks-and-defenses-for-federated-learning/)
- [FoolsGold](https://arxiv.org/abs/1808.04866)
- [CONTRA](https://par.nsf.gov/servlets/purl/10294585)
- [FLARE](https://dl.acm.org/doi/10.1145/3488932.3517395)
- [Romoa](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_23)
- [SignGuard](https://arxiv.org/abs/2109.05872)


All aggregation rules are located in _aggregation_rules.py_ as individual functions. 
To add an aggregation rule you can add the implementation in _aggregation_rules.py_.

## Attacks
To evaluate the robustness of the aggregation rules we also added the following attacks.

- [Label Flipping](https://proceedings.mlr.press/v20/biggio11.html)
- [Krum Attack](https://arxiv.org/abs/1911.11815)
- [Trim Attack](https://arxiv.org/abs/1911.11815)
- [Scaling Attack](https://arxiv.org/abs/2012.13995)
- [FLTrust Attack](https://arxiv.org/abs/2012.13995)
- [Min-Max Attack](https://par.nsf.gov/servlets/purl/10286354)
- [Min-Sum Attack](https://par.nsf.gov/servlets/purl/10286354)

The implementation of the attacks are all located in _attacks.py_ as individual functions.
To add a new attack the implementation can simply be added as a new function in this file.

## Models
We implemented a [GCN](https://github.com/petermchale/gnn) for regretion task.

The model is in a separate file in the _models_ folder of this project. 

To add models a new file containing a class that defines this classifier must be added.
Additionally, in _main.py_ the _get_net_ function needs to be expanded to enable the selection of this model.

## Datasets
We implemented the [ESOL](https://paperswithcode.com/dataset/esol-scaffold) dataset. 
It must be downloaded with the provided loading script in the _data_ folder.

Adding a new dataset requires adding the loading to the _load_data_ function in _data_loading.py_. 
This can either be simply done by adding an existing dataloader from PyTorch or requires custom data loading.
Additionally, the size of the data examples and the number of classes need to be added to the _get_shapes_ function to properly configure the model.
Furthermore, the _assign_data_ function needs to be extended to enable assigning the test and train data to the individual clients.

## Multi-Party Computation
To run the MPC Implementation the [code](https://github.com/data61/MP-SPDZ) for [MP-SPDZ](https://eprint.iacr.org/2020/521) needs to be downloaded separately using the installation script _mpc_install.sh_.
The following protocols are supported:

- [MASCOT](https://dl.acm.org/doi/abs/10.1145/2976749.2978357?casa_token=ANhMJsmbD9kAAAAA:uaMV-qJpBFxYVJZekcBq_wk7y7iCWyctOVlNzt30oWfT9Amh5uQG_D5NCb_SybJrV_90sTAcK00O) uses 2 or more parties in a malicious setting 
- Semi2k uses 2 or more parties in a semi-honest, dishonest majority setting
- [SPDZ2k](https://eprint.iacr.org/2018/482) uses 2 or more parties in a malicious, dishonest majority setting
- [Replicated2k](https://eprint.iacr.org/2016/768.pdf) uses 3 parties in a semi-honest, honest majority setting
- [PsReplicated2k](https://eprint.iacr.org/2019/164.pdf) uses 3 parties in a malicious, honest majority setting

# How to run?

The project can be simply cloned from git and then requires downloading the [ESOL](https://paperswithcode.com/dataset/esol-scaffold) dataset as described in the dataset section.

The project takes multiple command line arguments to determine the training parameters, attack, aggregation, etc. is used.
If no arguments are provided the project will run with the default arguments.
A description of all arguments can be displayed by executing:
```shell
python main.py -h
```
# Requirements
The project requires the following packages to be installed:

- Python 3.8.13 
- Pytorch 1.11.0
- Torchvision 0.12.0
- Numpy 1.21.5
- MatPlotLib 3.5.1
- HDBSCAN 0.8.28
- Perl 5.26.2

All requirements can be found in the _requirements.txt_.

# Credits
This project is based on code by Till Gehlhar et al. the authors of [SAFEFL](https://ieeexplore.ieee.org/abstract/document/10188630?casa_token=FDcsggxqcvwAAAAA:_JzLcQrYYbLfTwa_KSVOoy8iiDttXEeQ1y33HhvqEJl0BdmfaYHBXVS44Hx5IbdbRKNJTla8dg) and the github code is available [here](https://github.com/encryptogroup/SAFEFL)

We used the [open-sourced](https://github.com/encryptogroup/SAFEFL?tab=readme-ov-file) implementations of all agregation rules and all attacks.

The MPC Framework MP-SPDZ was created by [Marcel Keller](https://github.com/data61/MP-SPDZ).

# License
[]()
