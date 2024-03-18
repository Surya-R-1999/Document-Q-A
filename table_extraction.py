import os
import json
import glob
import cv2
import numpy as np
import requests
import base64
from pdf2image import convert_from_path
from ultralyticsplus import YOLO, render_result
from pdf2image import convert_from_path
import ast
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index import SimpleDirectoryReader
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
import sys
from pydantic import BaseModel
import pandas as pd
from joblib import Parallel, delayed

# Open AI Key
# open_ai_key =  "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"
val=0
###################################################################################################
# Prompts
###################################################################################################

# Defining the structure of the Metadata
class Tables(BaseModel):
    """Data model for a Tables."""
    # title: str
    # category: str
    # discount: str
    # price: str
    # rating: str
    # review: str
    # description: str
    # inventory: str
    title : str
    header : list
    # subheader : list
    rows : list
    description: str    
    
# OPENAI_API_TOKEN = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=10000
)

# try:
#     temp_path = str(sys._MEIPASS).replace('\\','/')
#     poppler_path = temp_path+'/poppler-23.11.0/Library/bin'
#     print('------> sys path\n',temp_path.split('/'))
# except:
#     cwd = os.getcwd()
#     user_dir = '/'.join(cwd.split('\\')[:3])
#     print('user dir -- > ', user_dir)
#     poppler_path=r"./poppler-23.11.0/Library/bin"

try:
    temp_path = str(sys._MEIPASS).replace('\\','/')
    poppler_path = temp_path+'/poppler-23.11.0/Library/bin'
    print('------> sys path\n',temp_path.split('/'))
except:
    cwd = os.getcwd()
    user_dir = '/'.join(cwd.split('\\')[:3])
    print('user dir -- > ', user_dir)
    poppler_path=f'{user_dir}/AppData/Local/anaconda3/envs/Doc2/Lib/site-packages/poppler-23.11.0/Library/bin'


    
###################################################################################################
# Prompts
###################################################################################################


# prompt_template_str = """
#     If the given image contain tables then understand the table and return the answer in json format else just provide None.
#         For example, if the image shows a table with the following headers and subheaders:
#         | System | Development | Test |
#         | EM% | EX% | EM% | EX% |

#         The output should be:

#         System | Development EM% | Development EX% | Test EM% | Test EX%

#         The rows should be in a list format, such as ['BRIDGE v2 + BERT', 71.1, 70.3, 67.5, 68.3].
# """

# prompt_template_str = """

#     Summarize the image, if it contains a table, otherwise provide None.
#     Don't generate random values if the table contains -   
    
#     """

# prompt_template_str = """
#     Summarize the image, if it contains a table, otherwise provide None.
    
#     Example 1:
    
#     If the image shows a table with the following headers and subheaders:
#     | System | Development | Test |
#     | EM% | EX% | EM% | EX% |

#     The output should be:

#     System | Development EM% | Development EX% | Test EM% | Test EX%

#     The rows should be in a list format and in the same order, such as ['BRIDGE v2 + BERT', 71.1, 70.3, 67.5, 68.3].

#     Please don't make assumptions and return the answer in JSON format.
#     """
    
prompt_template_str = """

    Summarize the image, if it contains a table, otherwise provide None.
    
    Use the below example's only if the Table_layout matches with the Table_layout of the table in Image. Otherwise summarize on your own.
    
    Example 1:
    
    If the image shows a table with the following headers and subheaders:
    
    Table_layout:
    
    ------------------------------------------------------------------------------------
    |                                                    	 Development		  Test	
    |                                                        ---------------------------
    |    System                                              EM%	 EX%	 EM%   	EX%
    ------------------------------------------------------------------------------------
    |BRIDGE v2 + BERT (ensemble) ¹ (Lin et al., 2020)	     71.1	70.3	67.5	68.3
    |SMBOP + GRAPPA (Rubin and Berant, 2021)	             74.7	75.0	69.5	71.1
    |RATSQL + GAP (Shi et al., 2021)	                     71.8	 -	    69.7	 -
    |DT-Fixup SQL-SP + ROBERTA (Xu et al., 2021)	         75.0	 -	    70.9	 -
    |LGESQL + ELECTRA (Cao et al., 2021)	                 75.1	 -	    72.0	 -
    |T5-Base (Shaw et al., 2021)	                         57.1	 -	     -	     -
    |T5-3B (Shaw et al., 2021)	                             70.0	 -	     -	     -
    |T5-Base (ours)	                                         57.2	57.9	 -	     -
    ------------------------------------------------------------------------------------
    |T5-Base+PICARD	                                         65.8	68.4	 -	     -
    |T5-Large	                                             65.3	67.2	 -	     -
    |T5-Large+PICARD	                                     69.1	72.9	 -	     -
    |T5-3B(ours)	                                         69.9	71.4	 -	     -
    |T5-3B+PICARD	                                         74.1	76.3	 -	     -
    |T5-3B                          	                     71.5	74.4	68.0	70.1
    |T5-3B+PICARD	                                         75.5	79.3	71.9	75.1
    ------------------------------------------------------------------------------------
    
    The rows should be in a list format and in the same order, represented in the output
    
    Output:
    
    header = ["System", "Development EM%", "Development EX%", "Test EM%" , "Test EX%"]
    
    
    rows = [[
    ["BRIDGE v2 + BERT (ensemble) ¹ (Lin et al., 2020)", 71.1, 70.3, 67.5, 68.3],
    ["SMBOP + GRAPPA (Rubin and Berant, 2021)", 74.7, 75.0,	69.5, 71.1],
    ["RATSQL + GAP (Shi et al., 2021)",	71.8, -, 69.7, -],
    ["DT-Fixup SQL-SP + ROBERTA (Xu et al., 2021)",	75.0, -	, 70.9,	-],
    ["LGESQL + ELECTRA (Cao et al., 2021)",	75.1, -	, 72.0,	-],
    ["T5-Base (Shaw et al., 2021)"	,57.1,	-,	-,	-],
    ["T5-3B (Shaw et al., 2021)",	70.0,	-	,-,	-],
    ["T5-Base (ours)",	57.2 ,	57.9,	-,	-],
    ["T5-Base+PICARD"	,65.8	,68.4,	-,	-],
    ["T5-Large",	65.3	,67.2,	-,	-]
    ["T5-Large+PICARD"	, 69.1,	72.9,	-	,-],
    ["T5-3B(ours)",	69.9	,71.4,	-,	-],
    ["T5-3B+PICARD"	,74.1,	76.3	,-	,-],
    ["T5-3B",	71.5,	74.4,	68.0,	70.1],
    ["T5-3B+PICARD",	75.5,	79.3,	71.9,	75.1]] 
    

    Example 2:
    
    Table_layout:
    
    ===========================================================================
    | Scaling | OPTQ | A-bits | W/V-bits | Hums | STEM | Social | Other | Avg. |
    ----------------------------------------------------------------------------
    |              Baseline              |61.90 | 52.10| 73.40  | 67.60 | 63.60|
    ----------------------------------------------------------------------------
    |    -    |  -   |        |          |54.10 | 46.40| 66.90  | 62.50 | 57.20|
    |    -    |  x   |        |          |56.00 | 47.80 | 67.20 | 63.90 | 58.50|
    -----------------|        |          |-------------------------------------|
    |   SQ	  |  x	 |  INT8  |  INT4    |57.70 | 47.50 | 67.90 | 64.30 | 59.30|
    -----------------|        |          |-------------------------------------| 
    |         |  x   |        |          |57.20 | 47.80 | 69.50 | 64.80 | 59.60| 
    |  AQAS   -------|        |----------|-------------------------------------|
    |         |  x   |        |  dINT4   |59.50 | 50.40 | 70.70 | 65.80 | 61.40| 
    ============================================================================
    
    Output:
    
    header = ["Scaling", "OPTQ", "A-bits", "W/V-bits", "Hums", "STEM", "Social", "Other", "Avg."]
    rows = [
    ["Baseline", "Baseline", "Baseline", "Baseline", "61.90"," 52.10", "73.40", "67.60", "63.60"],
    ["-", "-", "INT8", "INT4", "54.10", "46.40", "66.90", "62.50", "57.20"],
    ["-", "x", "INT8", "INT4", "56.00", "47.80", "67.20", "63.90", "58.50"],
    ["SQ", "x", "INT8", "INT4", "57.70", "47.50", "67.90", "64.30", "59.30"],
    ["AQAS", "x", "INT8", "dINT4", "57.20", "47.80", "69.50", "64.80", "59.60"],
    ["AQAS", "x", "INT8", "dINT", "59.50", "50.40", "70.70", "65.80", "61.40"]
]


    Example 3:
    
    Table_layout:
    
    ============================================================================    
    |  Model         |Precision	 |  W4 format   |Wikitext	PIQA	MMLU
    ----------------------------------------------------------------------------
    |                |     FP16	baseline	    |5.68	    78.29	35.20
    |                ------------------------------------------------------------
    |LLaMA-07B   	 |W4A16	     |FP4 (1-2-1)	|26.52	    62.84	27.31
    |                |           |FP4 (1-3-0)	|6.30	    76.77	31.46
    |                |           |  dINT4       |6.07	    77.91	32.53
    ----------------------------------------------------------------------------
    |                |     FP16	baseline	    |5.09	    78.78	47.15
    |                ------------------------------------------------------------
    |                |           |FP4 (1-1-2)	|74763.98	52.29	24.72
    |                | 		     |FP4 (1-2-1)	|7.95	    74.65	31.54
    |LLaMA-13B       |W4A16	     |FP4 (1-3-0)	|5.56	    78.62	40.76
    |                |           |  dINT4	    |5.38	    79.05	44.35
    ----------------------------------------------------------------------------
    |                |     FP16 baseline	    |4.10	    80.96	58.50
    |                ------------------------------------------------------------
    |                |           |FP4 (1-1-2)	|34027.07	51.52	25.32
    |LLaMA-30B 	     |W4A16	     |FP4 (1-2-1)	|9.10	    71.22	32.05
    |                |           |FP4 (1-3-0)	|4.57	    79.71	53.50
    |                |           |  dINT4	    |4.36	    80.41	55.87
    ============================================================================

    Output:

    header = [Model, Precision, W4 format, Wikitext, PIQA, MMLU]
    rows =   [["LLaMA-7B", "FP16	baseline" , "FP16	baseline", 5.68, 78.29, 35.20],
            ["LLaMA-7B", "W4A16", "FP4 (1-1-2)", 165582.55, 51.69, 26.88],
            ["LLaMA-7B", "W4A16", "FP4 (1-2-1)", 26.52, 62.84, 27.31],
            ["LLaMA-7B", "W4A16", "FP4 (1-3-0)", 6.30, 76.77, 31.46],
            ["LLaMA-7B", "W4A16", "dINT4", 6.07, 77.91, 32.53],
            ["LLaMA-13B", "FP16 baseline" , "FP16 baseline", 5.09, 78.78, 47.15],
            ["LLaMA-13B", "W4A16" , FP4 (1-1-2), 74763.98, 52.29, 24.72],
            ["LLaMA-13B", "W4A16" , "FP4 (1-2-1)", 7.95, 74.65, 31.54],
            ["LLaMA-13B", "W4A16" , "FP4 (1-3-0)", 5.56, 78.62, 40.76],
            ["LLaMA-13B", "W4A16" , "dINT4", 5.38, 79.05, 44.35],
            ["LLaMA-30B", "FP16 baseline", "FP16 baseline", 4.10, 80.96, 58.50],
            ["LLaMA-30B", "W4A16", "FP4 (1-1-2)", 34027.07, 51.52, 25.32],
            ["LLaMA-30B", "W4A16", "FP4 (1-2-1)", 9.10, 71.22, 32.05],
            ["LLaMA-30B", "W4A16","FP4 (1-3-0)", 4.57, 79.71, 53.50],
            ["LLaMA-30B", "W4A16", "dINT4",	4.36, 80.41	55.87]]


    Example 4:
    
    Table_layout:
    
    ============================================================
    |Weight  |  OPTQ  | W/V-bits|         OPT           | LLaMA |
    |Scaling |		 |			| -------------------------------
    |        |        |	        |  125M    2.7B   6.7B  |   7B  |
    |-------------------------------------------------------------
    |        Baseline	        |  31.95   14.32  12.29 |  5.68 |
    |------------------------------------------------------------
    | -     |    -   |   INT3   |  1.7e3   4.3e4  1.2e4 | 94.97 |
    | -     |    -   |  dINT3   | 127.94   8.7e3  55.24 | 10.99 |
    |-------------------------------------------------------------
    |       |    x   |   INT3   |  54.84   36.38  69.45 | 24.85 | 
    | AQAS  |    x   |  dINT3   |  46.34   20.67  17.42 | 10.04 |
    |-------------------------------------------------------------

    Output:

    header = [Weight Scaling, OPTQ, W/V-bits, OPT 125M, OPT 2.7B, OPT 6.7B, LLaMA]

    rows = [[Baseline,Baseline,Baseline,31.95,14.32,12.29,5.68],
        [-,-,INT3,1.7e3,4.3e4,1.2e4,94.97],
        [-,-,dINT3,127.94,8.7e3,55.24,10.99],
        [AQAS,x,INT3,54.84,36.38,69.45,24.85], 
        [AQAS,x,dINT3,46.34,20.67,17.42,10.04]]

    Please don't make assumptions and return the answer in JSON format.
    """


#############################################################################################
# ImageStore
#############################################################################################

def createImageStore(source_path):
    folder_path = source_path  # Specify the folder directory containing the PDF files
    image_store_folder = r"./DocData/ImageStore"  # Specify the parent image store folder path
    
    # Create the parent image store folder if it doesn't exist
    os.makedirs(image_store_folder, exist_ok=True)
    
    # Iterate through the PDF files in the folder directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            print(f"Processing ---> {file_name}")
            
            # Create the subfolder with the file name inside the image store folder
            subfolder_path = os.path.join(image_store_folder, os.path.splitext(file_name)[0])
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            else:
                # If subfolder already exists, move to the next file
                print("Images are already Existing")
                continue
            
            
            # Convert PDF to images and save them inside the subfolder
            source_path = os.path.join(folder_path, file_name)
            images = convert_from_path(source_path, poppler_path=poppler_path)
            for i, image in enumerate(images):
                image_path = os.path.join(subfolder_path, f"{os.path.splitext(file_name)[0]}_page{i}.jpg")
                image.save(image_path, 'JPEG')
            
        

    
    return True



################################################################################################
# TableStore
################################################################################################

def createTableStore(image_store):
    # Load model
    model = YOLO('keremberke/yolov8m-table-extraction')
    # Set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.47  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    
    # Directory paths
    image_store_folder = image_store
    table_store_folder = r"./DocData/TableStore"
    if not os.path.exists(table_store_folder):
        os.makedirs(table_store_folder)
        
    # tempdir_files = os.listdir(r'./DocData/tempdir')
    # Iterate through the subfolders in the ImageStore
    # for subfolder in os.listdir(image_store_folder):
    for subfolder in os.listdir(r'./DocData/tempdir'):
        base_name, extension = os.path.splitext(subfolder)
        subfolder_path = os.path.join(image_store_folder, base_name)
        if os.path.isdir(subfolder_path):
            print("Extracting Tables")
            # Check if the corresponding folder exists in TableStore
            table_subfolder_path = os.path.join(table_store_folder, base_name)
            if os.path.exists(table_subfolder_path):
                print("Table Folder already Existing")
                # If the folder exists in TableStore, skip processing and move to the next folder
                continue
            
            # Create corresponding subfolder in TableStore
            os.makedirs(table_subfolder_path)
            
            # Process each image in the subfolder
            pattern = "*.jpg"
            files = glob.glob(os.path.join(subfolder_path, pattern))
            for file in files:
                if os.path.isfile(file):  # Check if the file is a regular file
                    image = file
                    # Perform inference
                    results = model.predict(image)
                    # Observe results
                    # print(results[0].boxes)
                    render = render_result(model=model, image=image, result=results[0])
                    boxes = results[0].boxes.xyxy.tolist()
                    table_count = 1
                    for i in range(len(boxes)):
                        first_box = boxes[i]
                        table_cord = first_box[:4]
                        # table_cord[0] = table_cord[0] - 50
                        # table_cord[1] = table_cord[1] - 200
                        # table_cord[2] = table_cord[2] + 50
                        # table_cord[3] = table_cord[3] + 200
                        # print(table_cord)
                        cropped_img = np.asarray(render.crop(table_cord))
                        if cropped_img.dtype != np.uint8:
                            cropped_img = cropped_img.astype(np.uint8)
                        _, encoded_img = cv2.imencode('.jpg', cropped_img)
                        
                        # Get the file name and page number
                        file_name = os.path.splitext(os.path.basename(file))[0]
                        page_number = i
                        # Add "table" keyword to the file name
                        table_name = f"{file_name}_table{table_count}.jpg"
                        # Save the cropped image with the desired format
                        table_image_path = os.path.join(table_subfolder_path, table_name)
                        with open(table_image_path, 'wb') as f:
                            f.write(encoded_img)
                        table_count += 1
    return True


###################################################################################################
# Summarizing the Tables
###################################################################################################

def process_image(file, subfolder_path): #, json_data):
    # Check if text file already exists
    image_name = os.path.splitext(os.path.basename(file))[0]
    
    metadata_file = os.path.join(subfolder_path, f"{image_name}")
    
    # if metadata_file in json_data.keys():
    #     return metadata_file, None

    # Rest of your code to extract tables and create summary_dict
    try:
        # Code to extract tables from the image file
        table_image_documents = SimpleDirectoryReader(input_files=[file]).load_data()
        openai_program_amazon = MultiModalLLMCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(Tables),
            image_documents=[table_image_documents[0]],
            prompt_template_str=prompt_template_str,
            llm=openai_mm_llm,
            verbose=False,  # True
        )
        
        response = openai_program_amazon()
        # for res in response:
        #     print(res)

        # Extract page number and page name
        page_number = image_name.split("_")[1][4:]  # Remove the "page" prefix
        page_name = image_name.split("_")[0]

        if response is None:
            summary_dict = {
                "header": None,
                "rows": None,
                "page_no": page_number,
                "file_name": page_name,
                "image_path": file
            }
            
        else:
            # Create dictionary with summary, page number, page name, and image path
            summary_dict = {
                "header": response.header,
                "rows": response.rows,
                "description" : response.description,
                "page_no": page_number,
                "file_name": page_name,
                "image_path": file
            }

        return metadata_file, summary_dict

    except Exception as e:
        print(e)
        return metadata_file, None



###################################################################################################
# Parallel Computation for Summarizing the Tables
###################################################################################################

def metadataCreation(table_store_folder):
    
    # Specify the path to your JSON file
    
    json_summary_dir_path = "./DocData/SummaryJson"
    os.makedirs(json_summary_dir_path, exist_ok=True)
    json_files = os.listdir('./DocData/SummaryJson')
    
    #-------------------------------------------------------
    # json_file_path = r"./DocData/SummaryJson/metadata.json"
    # if not os.path.exists(json_file_path):
    #     temp_dict = {}
    #     with open(json_file_path, 'w') as fp:
    #         json.dump(temp_dict)
    # Read the JSON file
    # with open(json_file_path, "r") as f:
    #     json_data = json.load(f)
    #--------------------------------------------------------
    
        
    # Iterate through the subfolders in the TableStore
    # for subfolder in os.listdir(table_store_folder):
    for subfolder in os.listdir(r"./DocData/tempdir"):
        base_name, extension = os.path.splitext(subfolder)
        subfolder_path = os.path.join(table_store_folder, base_name)
            
        # Process each image in the subfolder
        pattern = "*.jpg"
        files = glob.glob(os.path.join(subfolder_path, pattern))
        
        
        if base_name+'.json' not in json_files:
            print(f"Creating metadata for {base_name}")
            
            # Perform parallel computation for image processing
            summaries = Parallel(n_jobs=8, prefer="threads")(
                delayed(process_image)(file, subfolder_path) for file in files
            )
            
            # Update the summary_log with the results
            summary_log = {metadata_file: summary_dict for metadata_file, summary_dict in summaries}
            
            with open(os.path.join(json_summary_dir_path,base_name+'.json'), 'w') as f:
                json.dump(summary_log, f)
            
        
        
    return json_summary_dir_path


###################################################################################################
# Main Function
###################################################################################################

def main(source_path = r"./DocData/SourceDirectory"):
    list_jsons = []
    try: 
        imageStoreStatus =  createImageStore(source_path)
        if imageStoreStatus == True:
            tableStoreStatus = createTableStore(image_store=r"./DocData/ImageStore")
            
            if tableStoreStatus == True:
                summary_folder_path = metadataCreation(table_store_folder=r"./DocData/TableStore")
            
        list_jsons = [i[:-4]+'.json' for i in os.listdir(source_path) if i.endswith('.pdf')]
    except Exception as e:
        print(e)
    
    return list_jsons


# result = main()

# print("List of Files Passed : ", result)


