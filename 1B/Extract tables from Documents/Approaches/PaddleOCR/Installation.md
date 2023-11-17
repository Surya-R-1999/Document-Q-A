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

![img_12](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/886dbc9e-2a9b-42cb-9138-c9dee43c61de)


- move to the directory where the folder containing images,
```
cd /path/to/ppocr_img
(paddle_env) E:\Surya\Document-Q-A\1B\Extract tables from Documents\Approaches\paddleOCR\ppocr_img\ppocr_img>paddleocr --image_dir ./table/result_all.jpg --use_angle_cls true --lang en --use_gpu false
```
- O/P:

```
(paddle_env) E:\Surya\Document-Q-A\1B\Extract tables from Documents\Approaches\paddleOCR\ppocr_img\ppocr_img>paddleocr --image_dir ./imgs_en/img_12.jpg --use_angle_cls true --lang en --use_gpu false
download https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar to C:\Users\FUT_Surya/.paddleocr/whl\det\en\en_PP-OCRv3_det_infer\en_PP-OCRv3_det_infer.tar
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.00M/4.00M [00:06<00:00, 609kiB/s]
download https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar to C:\Users\FUT_Surya/.paddleocr/whl\rec\en\en_PP-OCRv4_rec_infer\en_PP-OCRv4_rec_infer.tar
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10.2M/10.2M [00:21<00:00, 480kiB/s]
download https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar to C:\Users\FUT_Surya/.paddleocr/whl\cls\ch_ppocr_mobile_v2.0_cls_infer\ch_ppocr_mobile_v2.0_cls_infer.tar
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.19M/2.19M [00:11<00:00, 188kiB/s]
[2023/11/15 08:54:39] ppocr DEBUG: Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls_batch_num=6, cls_image_shape='3, 48, 192', cls_model_dir='C:\\Users\\FUT_Surya/.paddleocr/whl\\cls\\ch_ppocr_mobile_v2.0_cls_infer', cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det=True, det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, det_limit_side_len=960, det_limit_type='max', det_model_dir='C:\\Users\\FUT_Surya/.paddleocr/whl\\det\\en\\en_PP-OCRv3_det_infer', det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', e2e_char_dict_path='./ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_id=0, gpu_mem=500, help='==SUPPRESS==', image_dir='./imgs_en/img_12.jpg', image_orientation=False, invert=False, ir_optim=True, kie_algorithm='LayoutXLM', label_list=['0', '180'], lang='en', layout=True, layout_dict_path=None, layout_model_dir=None, layout_nms_threshold=0.5, layout_score_threshold=0.5, max_batch_size=10, max_text_length=25, merge_no_span_structure=True, min_subgraph_size=15, mode='structure', ocr=True, ocr_order_method=None, ocr_version='PP-OCRv4', output='./output', page_num=0, precision='fp32', process_id=0, re_model_dir=None, rec=True, rec_algorithm='SVTR_LCNet', rec_batch_num=6, rec_char_dict_path='C:\\Users\\FUT_Surya\\AppData\\Local\\anaconda3\\envs\\paddle_env\\lib\\site-packages\\paddleocr\\ppocr\\utils\\en_dict.txt', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_model_dir='C:\\Users\\FUT_Surya/.paddleocr/whl\\rec\\en\\en_PP-OCRv4_rec_infer', recovery=False, save_crop_res=False, save_log_path='./log_output/', scales=[8, 16, 32], ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ser_model_dir=None, show_log=True, sr_batch_num=1, sr_image_shape='3, 32, 128', sr_model_dir=None, structure_version='PP-StructureV2', table=True, table_algorithm='TableAttn', table_char_dict_path=None, table_max_len=488, table_model_dir=None, total_process_num=1, type='ocr', use_angle_cls=True, use_dilation=False, use_gpu=False, use_mp=False, use_npu=False, use_onnx=False, use_pdf2docx_api=False, use_pdserving=False, use_space_char=True, use_tensorrt=False, use_visual_backbone=True, use_xpu=False, vis_font_path='./doc/fonts/simfang.ttf', warmup=False)
[2023/11/15 08:54:41] ppocr INFO: **********./imgs_en/img_12.jpg**********
[2023/11/15 08:54:41] ppocr DEBUG: dt_boxes num : 11, elapsed : 0.5993988513946533
[2023/11/15 08:54:41] ppocr DEBUG: cls num  : 11, elapsed : 0.14373779296875
[2023/11/15 08:54:46] ppocr DEBUG: rec_res num  : 11, elapsed : 4.954953908920288
[2023/11/15 08:54:46] ppocr INFO: [[[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]], ('ACKNOWLEDGEMENTS', 0.9974855184555054)]
[2023/11/15 08:54:46] ppocr INFO: [[[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]], ('We would like to thank all the designers and', 0.968330979347229)]
[2023/11/15 08:54:46] ppocr INFO: [[[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]], ('contributors who have been involved in the', 0.9776102900505066)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 446.0], [1207.0, 443.0], [1208.0, 484.0], [399.0, 488.0]], ('production of this book; their contributions', 0.9866492748260498)]
[2023/11/15 08:54:46] ppocr INFO: [[[401.0, 500.0], [1208.0, 500.0], [1208.0, 534.0], [401.0, 534.0]], ('have been indispensable to its creation.We', 0.9628525972366333)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 550.0], [1209.0, 548.0], [1209.0, 583.0], [399.0, 584.0]], ('would also like to express our gratitude to all', 0.9740485548973083)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 600.0], [1207.0, 598.0], [1208.0, 634.0], [399.0, 636.0]], ('the producers for their invaluable opinions', 0.9963331818580627)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 648.0], [1207.0, 646.0], [1208.0, 686.0], [399.0, 688.0]], ('and assistance throughout this project. And to', 0.9943731427192688)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 702.0], [1209.0, 698.0], [1209.0, 734.0], [399.0, 738.0]], ('the many others whose names are not credited', 0.977229118347168)]
[2023/11/15 08:54:46] ppocr INFO: [[[399.0, 750.0], [1211.0, 750.0], [1211.0, 789.0], [399.0, 789.0]], ('but have made specific input in this book, we', 0.9979288578033447)]
[2023/11/15 08:54:46] ppocr INFO: [[[397.0, 802.0], [1090.0, 800.0], [1090.0, 839.0], [397.0, 841.0]], ('thank you for your continuous support.', 0.9981997609138489)]

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
O/P:
```
[[397.0, 802.0], [1092.0, 802.0], [1092.0, 841.0], [397.0, 841.0]]
[[397.0, 750.0], [1211.0, 750.0], [1211.0, 789.0], [397.0, 789.0]]
[[397.0, 702.0], [1209.0, 698.0], [1209.0, 734.0], [397.0, 738.0]]
......
```
- Only Recognition: set --det to false
```
paddleocr --image_dir ./imgs_words_en/word_10.png --det false --lang en
```
O/P:
```
['PAIN', 0.9934559464454651]
```
- PaddleOCR currently supports 80 languages







