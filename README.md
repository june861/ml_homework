## 北邮高级机器学习课后作业

克隆仓库到本地，然后安装ml_homework库
```bash
git clone git@github.com:june861/ml_homework.git
cd ml_homework
pip install -e .
```
作业A1 - 基于MLP的CIFAR10图像分类
```bash
# 确保目录在ml_homework中
cd ml_homework
bash scripts/scripts/A1.sh
```
作业A2 - 基于CNN的CIFAR10图像分类
```bash
cd ml_homework
bash scripts/scripts/A2.sh
```


项目目录结构如下：
```bash
.
├── README.md
├── data
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── logs
├── ml_homework.egg-info
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
├── requirements.txt
├── result
├── runs
├── scripts
│   ├── learners
│   │   ├── __pycache__
│   │   │   ├── a1_learner.cpython-39.pyc
│   │   │   ├── a2_learner.cpython-39.pyc
│   │   │   └── base_learners.cpython-39.pyc
│   │   ├── a1_learner.py
│   │   ├── a2_learner.py
│   │   └── base_learners.py
│   ├── mains
│   │   ├── a1_main.py
│   │   └── a2_main.py
│   └── scripts
│       ├── A1.sh
│       └── A2.sh
├── setup.py
├── utils
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-39.pyc
│   │   ├── cnn.cpython-39.pyc
│   │   ├── config.cpython-39.pyc
│   │   ├── dataset.cpython-39.pyc
│   │   ├── mlp.cpython-39.pyc
│   │   └── utils.cpython-39.pyc
│   ├── cnn.py
│   ├── config.py
│   ├── dataset.py
│   ├── mlp.py
│   └── utils.py
└── wandb
```

