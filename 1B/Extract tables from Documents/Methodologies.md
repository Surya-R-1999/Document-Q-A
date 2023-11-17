# Methods to Extract Table's from Documents :

- Extracting tables from multiple format documents and storing them in a CSV file can be a challenging task, as different types of documents may have different structures and formats. However, various methods and tools are available to help automate this process. In this passage, we will discuss the performance of four methods in extracting tables from documents of different formats and storing them in a CSV file.

- The four methods tested in this passage are PDFPlumber, PaddleOCR, YOLO, and three others. PDFPlumber is a tool that uses Optical Character Recognition (OCR) technology to extract tables from PDF documents. PaddleOCR is an open-source OCR engine that can extract tables from various formats, including PDF, TXT, and Images. YOLO is a deep learning-based method that uses object detection to locate and extract tables from documents.

- The results of the experiment show that PDFPlumber was able to detect and recognize the tables in the documents and store them in a CSV file with high accuracy. PaddleOCR, on the other hand, used NLP models to extract the texts with an accuracy of 90%. YOLO was able to detect the tables and provided the coordinates, but it did not store the tables in a CSV file.

- Overall, these results demonstrate the effectiveness of different methods in extracting tables from documents of different formats and storing them in a CSV file. PDFPlumber and PaddleOCR are suitable for extracting tables from PDF documents, while YOLO can be used for detecting tables in images. These methods can be used in various industries, such as finance, healthcare, and education, where automated table extraction is a crucial task.

- In conclusion, extracting tables from multiple format documents and storing them in a CSV file is a task that can be automated using various methods and tools. The choice of method depends on the format of the documents and the desired level of accuracy. By using these methods, organizations can save time and reduce manual effort in data entry, which can lead to increased productivity and accuracy.

**1. Table Transformer (TATR)**
- source : https://github.com/microsoft/table-transformer#table-transformer-tatr
-  A deep learning model based on object detection for extracting tables from PDFs and images.
-  First proposed in "PubTables-1M: Towards comprehensive table extraction from unstructured documents".

![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/66c9dacc-04af-4230-94da-2290860fe095)

This repository also contains the official code for these papers:

"GriTS: Grid table similarity metric for table structure recognition"
"Aligning benchmark datasets for table structure recognition"
Note: If you are looking to use Table Transformer to extract your own tables, here are some helpful things to know:

TATR can be trained to work well across many document domains and everything needed to train your own model is included here. But at the moment pre-trained model weights are only available for TATR trained on the PubTables-1M dataset. (See the additional documentation for how to train your own multi-domain model.)
TATR is an object detection model that recognizes tables from image input. The inference code built on TATR needs text extraction (from OCR or directly from PDF) as a separate input in order to include text in its HTML or CSV output.

**2. pdfplumber**
- source : https://github.com/jsvine/pdfplumber
- Plumb a PDF for detailed information about each text character, rectangle, and line. Plus: Table extraction and visual debugging.
-  Works best on machine-generated, rather than scanned, PDFs. Built on pdfminer.six.
- Currently tested on Python 3.8, 3.9, 3.10, 3.11.

![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/d269fd71-ff28-456a-9347-aa5c84ed4dbe)

**3. Multi-Type-TD-TSR**
- source : https://github.com/Psarpei/Multi-Type-TD-TSR
- Extracting Tables from Document Images using a Multi-stage Pipeline for Table Detection and Table Structure Recognition
  
![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/7e3f4288-38e8-4c3c-9a51-3c7404ebdb99)

**4. img2table (Multi Page Images Not Supported)**
- source :  https://github.com/xavctn/img2table
- img2table is a simple, easy to use, table identification and extraction Python Library based on OpenCV image processing that supports most common image file formats as well as PDF files.

**5. CascadeTabNet**
- source : https://github.com/DevashishPrasad/CascadeTabNet
- CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents
- The paper was presented (Orals) at **CVPR 2020 Workshop on Text and Documents in the Deep Learning Era**
- Table Detection:
  
![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/ac19c64c-aa3f-407b-80ce-451a89ca4d99)
  
- Table Structure Recognition:(Bordered Table and Borderless table)
  
![image](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/379edc57-7172-4fec-ba89-610a9618bce6)

**6. PaddleOCR**

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

![img_12](https://github.com/Surya-R-1999/Document-Q-A/assets/121089254/2f4d79d6-0ee8-4d6c-87ed-523f5a72ba2a)



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







