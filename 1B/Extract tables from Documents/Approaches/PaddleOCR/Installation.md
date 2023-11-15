# PaddleOCR

- Create a new Conda environment
```
conda create --name paddle_env python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```
- To activate the Conda environment,
```
conda activate paddle_env
```
- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
```
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- If you have no available GPU on your machine, please run the following command to install the CPU version
```
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- Install PaddleOCR Whl Package
```
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
```
