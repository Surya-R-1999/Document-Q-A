# PaddleOCR

- Create a new Conda environment
```
conda create --name paddle_env python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
```
- To activate the Conda environment,
```
conda activate paddle_env
```

- PreRequisite: Install CUDA 10
   
- If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
```
python -m pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- If you have no available GPU on your machine, please run the following command to install the CPU version
```
python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- Install PaddleOCR Whl Package (# Recommend to use version 2.0.1+)
```
pip install "paddleocr>=2.0.1" 
```

# Command Line Implementation:

- move to the directory where the folder containing images,
```
cd /path/to/ppocr_img
(paddle_env) E:\Surya\Document-Q-A\1B\Extract tables from Documents\Approaches\paddleOCR\ppocr_img\ppocr_img>paddleocr --image_dir ./table/result_all.jpg --use_angle_cls true --lang en --use_gpu false
```
- --image_dir is a parameter, pass the path of image in image_dir

- pdf file is also supported,
  
```
paddleocr --image_dir ./xxx.pdf --use_angle_cls true --use_gpu false --page_num 2
```
- pass the path of the pdf file to the image directory and by changing the page number we can detect and recognize the text, images in the corresponding page

- Only detection: set --rec to false

```
paddleocr --image_dir ./imgs_en/img_12.jpg --rec false

```






