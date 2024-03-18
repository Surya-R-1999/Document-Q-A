# for image extraction
import cv2
from  matplotlib import pyplot as plt
from pdf2image import convert_from_path
import numpy as np
import os
import sys
import datetime
import zipfile
from PIL import Image
from numpy import asarray
import json    
# import docx2txt

# for image summarization
import openai
# from openai.error import Timeout
import base64
import requests
from joblib import Parallel,delayed
from functools import partial
from tqdm import tqdm

# using Llama Index
from pydantic import BaseModel
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index import SimpleDirectoryReader
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from langchain.docstore.document import Document




prompt_template_str = """\
    if the given image contain graphs then understand each graph and return the answer in json format.
    Even if it is a normal image summarize what is in the image\
    and return the answer with json format \
"""


## OPEN AI KEY
os.environ['OPENAI_API_KEY'] =  ''
open_ai_key =  None

try:
    temp_path = str(sys._MEIPASS).replace('\\','/')
    poppler_path = temp_path+'/poppler-23.11.0/Library/bin'
    print('------> sys path\n',temp_path.split('/'))
except:
    cwd = os.getcwd()
    user_dir = '/'.join(cwd.split('\\')[:3])
    print('user dir -- > ', user_dir)
    poppler_path=f'{user_dir}/AppData/Local/anaconda3/envs/Doc2/Lib/site-packages/poppler-23.11.0/Library/bin'

########################################################
#### Image Extraction

############# For All Zippable File Formats ############

def crop_images_from_Ms_formats(filePath):
    """
    extracts images/figures/graphs from docx/ppt/excel/csv 
    input:
        filePath = input file path as string
    return:
        directory path to the extracted images
    """

    filename = filePath.split('\\')[-1] if '\\' in filePath else filePath.split('/')[-1]
    filename = filename.replace('.','_')

    # filename = filePath.split('\\')[-1].replace('.','_')
    new_dir = f"./DocData/cropped_images/{filename}"

    try:
        # creating a new dir to store images for the give doc
        # if the dir is already there throws an error, which means the doc is already processed 
        # or the different doc with an existing file name ( in the case the doc name must be chaged by the user )
        os.makedirs(new_dir)

        zipf = zipfile.ZipFile(filePath)

        filelist = zipf.namelist()

        for i,fname in enumerate(filelist):
            _, extension = os.path.splitext(fname)
            if extension in [".jpg", ".jpeg", ".png", ".bmp"]:
                dst_fname = f"{new_dir}/{os.path.basename(fname)}"
                # dst_fname = f"{new_dir}/{fname}"
                with open(dst_fname, "wb") as dst_f:
                    dst_f.write(zipf.read(fname))

    except FileExistsError as e:
        # print('DirError --> ',e)
        print(f'Processed File, Images are available at  {new_dir} !!!')

    except Exception as e:
        print('Error in crop_images_from_Ms_formats --> \n', e)
    
    return new_dir


############## DOCX ##################

# def get_images_from_docx(input_loc):
    
#     docxFile = input_loc.split('\\')[-1] if '\\' in input_loc else input_loc.split('/')[-1]
#     docxFile = docxFile.replace('.','_')

#     os.makedirs(f"./cropped_images/{docxFile}", exist_ok=True)
#     output_loc =  f"./cropped_images/{docxFile}/"
    
#     text = docx2txt.process(input_loc, output_loc)


############### PDF ###################
def get_page_images_from_pdf(path, poppler_path=poppler_path):
    images = convert_from_path(path,
                                poppler_path=poppler_path)
    return images


# Image padding parameters (in pixels): Set padding values to bring the text associated to your image,
# This is particular useful with visuals/charts without boarders, and you like to consider the axis values, 
# image title, or image descriptions as part of the visual.

MINIMUM_WIDTH = 0.07
MINIMUM_HEIGTH = 0.07
LEFT_PADDING = 60 #100
RIGHT_PADDING = 60 # 10
TOP_PADDING = 20 #5
BOTTOM_PADDING = 100 #280



# the function will crop the figures from the given page image based on edge detection - contours
def get_cropped_images(page_image):

    #### - Convert the page image to a bytearray readable by OpenCV
    original_img = cv2.cvtColor(np.asarray(page_image), code=cv2.COLOR_RGB2BGR)

    minimum_width = int(original_img.shape[1] * MINIMUM_WIDTH)
    minimum_height = int(original_img.shape[0] * MINIMUM_HEIGTH)

    #### - Iterate over the contours list to crop and redact from the original document the visuals found in the canny edged image
    #Iterate over the contours detected

    # Convert Page to gray scale
    gray_scale_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    #### - Detect edges using OpenCV Canny Edge detector on grayscale image
    # Apply a Canny Edge detector to the gray scale page and detect edges in the gray scale page
    canny_img = cv2.Canny(gray_scale_image, 0, 255, apertureSize=5, L2gradient=True)   

    #Detect the image contours1 in the edges detected
    contours1, hierarchy = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #### - Set the minimum width and height for filtering the contours1 size
    #Set the minimun dimensions for the images inside the page
    crp_imgs = []
    crp_imgs2 = []
    for c in contours1:        
        #Get the contour corner (x,y) width and height
        x,y,w,h = cv2.boundingRect(c)

        #Verify if the contour dimensions match the minimun dimensions set with minimum_width and minimum_height
        if (w >= minimum_width and h >= minimum_height):
            crp_imgs.append([ x,y,w,h])  # cropped image without paddings 
            crp_imgs2.append([ x - LEFT_PADDING, y - TOP_PADDING,   w + RIGHT_PADDING, h + BOTTOM_PADDING])   # cropped image with padding
            # cv2.imwrite("./crpd_00824.png",cropped_image)
            # cropped_image = None

    return crp_imgs2, original_img


### Merging near by rectangles 

# merging 2 images
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]


# checking the intersection of images
def _intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if h < 0:               # in original code :  if w<0 or h<0:
        return False
    return True


# find the rectangle that are close enough to merge
def _group_rectangles(rec):
    """
    Uion intersecting rectangles.
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles 

    ### Reference for image intersection checking and merging 
    ### https://github.com/Breta01/handwriting-ocr/blob/master/src/ocr/words.py
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        # print(i)
        if not tested[i]:
            j = i+1
            while j < len(rec):
                if not tested[j] and _intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


def crop_n_save_images_from_pdf(filePath):
    """
    crops images/figures/graphs from pdf files
    input:
        filePath = pdf file path as string
    return:
        directory path to the cropped images
    """

    t1 = datetime.datetime.now()

    filename = filePath.split('\\')[-1] if '\\' in filePath else filePath.split('/')[-1]
    filename = filename.replace('.','_')
    print('Processing file ---> ', filename)

    # created a directory to store cropped image of a pdf
    new_dir = f"./DocData/cropped_images/{filename}"

    try:
        # creating a new dir to store images for the give doc
        # if the dir is already there throws an error, which means the doc is already processed 
        # or the different doc with an existing file name ( in the case the doc name must be chaged by the user )
        os.makedirs(new_dir)

        images = get_page_images_from_pdf(filePath)
        cropped_image_count = 0

        for indx, img in enumerate(images):
            crp_imgs, original_img = get_cropped_images(img)

            if len(crp_imgs) > 0:
                boundingBoxes = _group_rectangles(crp_imgs)

                for num, fig in enumerate(boundingBoxes):
                    [x, y, w, h] = fig
                    cv2.imwrite(f"{new_dir}/page_{indx+1}_fig_{num+1}.png", original_img[y :(y + h + 100), x-20 :(x + w + 50)])
                    cropped_image_count += 1

        print(f'Total images saved for file {filename} --->', cropped_image_count)

    except FileExistsError as e:
        # print('DirError --> ',e)
        print(f'Processed File, Images are available at {new_dir} !!!')
    
    except Exception as e:
        print('Error in crop_n_save_images_from_pdf --> \n', e)

    t2 = datetime.datetime.now() - t1
    print('time of execution --> ', t2)

    return new_dir



################################################
#### Summarization

# Function to encode the image
def get_image_encoding(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_array(image_path):

    # load the image
    img = Image.open(image_path)
    
    # PIL images into NumPy arrays
    numpydata = asarray(img)

    return numpydata
    

def get_image_summary(image_path):
    """
    Summarizes the images given
    input:
        image_path : image file path as a string
        api_key : openai api key as a string
    return:
        summary : summary of the image as a string
    """

    # Getting the base64 string
    base64_image = get_image_encoding(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": """Your are a research assistant, your a given an image/graph/figure/table from a research paper or a document. 
                            You need understand and provide title and small discription about that image/graph/figure/table.
                            If it is a graph please try to provide information about the legends.
                            """
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 300
    }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        summary = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        print(response)
        summary = ''

    return image_path, summary


def generate_n_save_img_summaries(folder_path):
    """
    Takes images from a folder, summarizes, 
        save the summary in json file in images_summaries folder
    input:
        folder_path : folder path, as string, that contains the images  
    return:
        curr_json_file_path : json file path that contains the summaries for all the images in the doc
    """
    images = os.listdir(folder_path)

    filename = folder_path.split('\\')[-1] if '\\' in folder_path else folder_path.split('/')[-1]
    filename = filename.replace('.','_')

    img_summary_folder = f'./DocData/images_summaries'
    os.makedirs(img_summary_folder, exist_ok=True)

    image_json_files = os.listdir(img_summary_folder)
    curr_json_file_path = f'{img_summary_folder}/{filename}.json'

    image_dict = {}

    # parallel execution - summary gen
    if curr_json_file_path not in image_json_files  and len(images) > 0:
        summaries = []
        try:
            summaries = Parallel(n_jobs=8,prefer="threads")( delayed( partial(get_image_summary) ) (os.path.join(folder_path,image)) for image in tqdm( images ))
        except Exception as e:
            print("Error in parallel execution - 1 \n")
            print(e)
        finally:
            for i,img in enumerate(summaries): 
                image_name = img[0].split('\\')[-1] if '\\' in img[0] else img[0].split('/')[-1]
                image_dict[image_name] = {'image_description':img[-1], 'image_path':img[0]}
                print(image_dict)

        with open(curr_json_file_path, 'w') as fp:
            json.dump(image_dict, fp)
    # sequential execution - summary gen
    # if curr_json_file_path not in image_json_files  and len(images) > 0:
    #     for image in images:
                
    #         try:
    #             img_path = os.path.join(folder_path,image)
    #             _, img_summary = get_image_summary(img_path)
    #             # img_arr = get_image_array(img_path)
    #             # print(img_path, img_summary, img_arr)
    #             image_dict[image] = {'summary':img_summary, 'image_path':img_path}#"image_array":img_arr.tolist()}
                
    #         except Exception as e:
    #             print('Error in image summary generation \n', e)


    else:
        print("Json Data is available ......") if len(images) > 0 else print("No images in given Document")
        print(curr_json_file_path,'\n')



    return curr_json_file_path




###### Using LLama Index ###########

class DocImage(BaseModel):
    """Data model for an Image/Graph/Figure from a Research Document."""

    Title: str
    Image_Description: str
    x_axis: str
    y_axis: str
    legend: str or list
    trend: str

    def to_json(self):
        json_obj = {}
        json_obj["Title"] = self.Title
        json_obj["Image_Description"] = self.Image_Description
        json_obj["x_axis"] = self.x_axis
        json_obj["y_axis"] = self.y_axis
        json_obj["legend"] = self.legend
        json_obj["trend"] = self.trend
        return json_obj

class ListImages(BaseModel):
    Figure : list[DocImage]

    def get(self):
        return self.Figure
    
# put your local directory here
def get_LLamaImages_from_LocalDir(image_dir_path):
    image_documents = []
    if len(os.listdir(image_dir_path)) > 0:
        image_documents = SimpleDirectoryReader(image_dir_path).load_data()
    return image_documents

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", 
    api_key=os.environ['OPENAI_API_KEY'], 
    max_new_tokens=5000
)


def get_LLamaImage_Summary(imageDoc):
    """
    Summarizes the images given and returns the summary in a document with metadata
    input:
        imageDoc : image in Document format
    return:
        summary_Document : summary of the image in Langchain document format
    """

    imgPath = imageDoc.dict()['metadata']['file_path']

    openai_program = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(ListImages),
    image_documents=[imageDoc],
    prompt_template_str=prompt_template_str,
    llm=openai_mm_llm,
    verbose=False,  # True
    )

    response = openai_program()

    summary_string = ''
    summary_dict = {}
    for i, res in enumerate(response.get()):
        # summary_string += str(res.to_json()) + '\n'
        summary_dict[f'Figure{i}'] = res.to_json()
    
    doc =  Document(page_content=summary_string, 
                metadata={
                    "image_path": imgPath, 
                    "file_name": imgPath.split('\\')[-1] })
    
    return  imgPath, summary_dict



def generate_n_save_LLama_img_summaries(folder_path):
    """
    Takes images from a folder, summarizes,  
        save the summary and arrya in json file in images_summaries folder
    input:
        folder_path : folder path, as string, that contains the images  
    return:
        curr_json_file_path : json file path that contains the summaries  for all the images in the doc
    """
    # images = os.listdir(folder_path)
    image_documents = get_LLamaImages_from_LocalDir(folder_path)

    filename = folder_path.split('\\')[-1] if '\\' in folder_path else folder_path.split('/')[-1]
    filename = filename.replace('.','_')

    img_summary_folder = './DocData/LLama_images_summaries'
    os.makedirs(img_summary_folder, exist_ok=True)

    image_json_files = os.listdir(img_summary_folder)
    curr_json_file_path = f'{img_summary_folder}/{filename}.json'

    image_dict = {}

    # parallel execution - summary gen
    if curr_json_file_path.split('/')[-1] not in image_json_files  and len(image_documents) > 0:
        print(curr_json_file_path.split('/')[-1])
        summaries = []
        try:
            print('--- Image Summarization - running Parallel ---')
            summaries = Parallel(n_jobs=8,prefer="threads")( delayed( partial(get_LLamaImage_Summary) ) (image) for image in tqdm(image_documents ))
        except Exception as e:
            print("Error in parallel execution - 2 \n")
            print(e)
        finally:
            for i,img in enumerate(summaries):
                path, summary_dict = img 
                image_name = path.split('\\')[-1] if '\\' in path else path.split('/')[-1]
                image_dict[image_name] = {'image_path':path}
                image_dict[image_name].update(summary_dict)

            with open(curr_json_file_path, 'w') as fp:
                json.dump(image_dict, fp)
    else:
        print("Json Data is available ......") if len(image_documents) > 0 else print("No images in given Document")
        print(curr_json_file_path,'\n')


    return curr_json_file_path




############ The image processing pipeline

def image_process_pipeline(document_path, api_key = None):
    """
    Runs the document image processing pipeline, includes the following steps
        - Extract/Crop images from pdfs/docx/ppts/excel/csv
        - save those cropped images in a folder 
        - summarize the images and convert them into numpy arrays
        - save these summaries and ndarrays in a json

    input:
        document_path : document path as string 
        api_key : openai api key as string
    return:
        curr_json_file_path : json file path that contains the summaries and ndarrays for all the images in the doc
    """
    summary_json_file = ''
    t1 = datetime.datetime.now()

    # check openai key is given or available in env 
    if api_key != None: os.environ['OPENAI_API_KEY'] = api_key 
    elif os.environ['OPENAI_API_KEY'] : print("-- OpenAi API Key is not provided -- ")


    crpd_imgs_dir_path = ""
    filename = document_path.split('\\')[-1] if '\\' in document_path else document_path.split('/')[-1]
    ext = document_path.split('.')[-1]

    print('----- Doc Image Extraction -----')

    if ext == 'pdf':
        crpd_imgs_dir_path = crop_n_save_images_from_pdf(document_path)

    elif ext == 'msg':
        filename = filename.replace('.','_')
        crpd_imgs_dir_path = f"./DocData/cropped_images/{filename}"

    elif ext in ['tiff', 'tif', 'jpeg', 'png', "jpg", 'jfif']:
        image_tiff = Image.open(document_path)
        filename = filename.replace('.','_')
        crpd_imgs_dir_path = f"./DocData/cropped_images/{filename}" 
        image_tiff.save(f"{crpd_imgs_dir_path}/{filename}.png")

    elif ext in ['xlsx','xls', 'csv','docx','ppt','pptx','pptm']:
        crpd_imgs_dir_path = crop_images_from_Ms_formats(document_path)

    else:
        print('Document with this extention can not be processed :( ')
        # crpd_imgs_dir_path = None

    if os.path.exists(crpd_imgs_dir_path):
        print(f'cropped images for {document_path}, available at {crpd_imgs_dir_path}')
        print('----- Doc Image Summarization -----')
        # summary_json_file = generate_n_save_img_summaries(crpd_imgs_dir_path)
        summary_json_file = generate_n_save_LLama_img_summaries(crpd_imgs_dir_path)

    t2 = datetime.datetime.now() - t1
    print("Total Pipeline Execution time : ", t2)

    return summary_json_file
    



# """
# Your are a research assistant, your a given an image/graph/figure/table from a research paper or a document. 
#                             You need understand and provide the detailed summary of it.
#                             if ther is a grpah give point wise observations, trends and detailed analysis of the graph.
# """