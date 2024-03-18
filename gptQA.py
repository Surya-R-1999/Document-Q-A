import openai
from openai import OpenAI
from configparser import ConfigParser
# for image extraction
from pdf2image import convert_from_path
import os
import sys
import json

# for image summarization
import openai
import requests
import base64
# from spire.doc import *
import spire.doc
from spire.doc.common import *
# from spire.presentation import *
import spire.presentation
from spire.presentation.common import *


#Read config.ini file
# config_object = ConfigParser()
# config_object.read("./DocConfig.config")
# userinfo = config_object["Keys"]
# os.environ["OPENAI_API_KEY"] = userinfo['open_ai_key']
# open_ai_key = userinfo['open_ai_key']
#Read config.ini file
config_object = ConfigParser()
#Read config.ini file
try:
    temp_path = str(sys._MEIPASS).replace('\\','/')
    configPath = temp_path+'/DocQA/DocConfig.config'
    print('------> sys path\n',configPath.split('/'))
    config_object.read(configPath)
except:
    config_object.read("./DocConfig.config")

userinfo = config_object["Keys"]
os.environ["OPENAI_API_KEY"] = userinfo['open_ai_key']
open_ai_key = userinfo['open_ai_key']


def get_image_encoding(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


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
                "text": ''' your are a Research Assistant. Your are given an image of a first page of a research paper. 
                            Identify the Title of the that paper and return just the title. 
                        '''
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
        "max_tokens": 600
    }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    try:
        summary = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        print(response)
        summary = ''

    return image_path, summary

def get_class(answer, MaxToken=400, outputs=3): 

    PROMPT = f'''
        your are a Research Assistant. 
        you are given a text, which is answer for a question, you need classify it as negative response or positive response.
        The text must not be like an assistant asking for more information or an assistant saying that it can not answer a question, if the 
        text is like that then consider it as negative.
        Only if the text contains some summary or some explanation of a topic then only consider it as positive.
        If it is negative response return 0, if it is positive response return 1.
        
        example 1:
            text : To determine what the document is about, I would need to analyze the content of the document. Please provide the document or specify the tool you would like me to use to analyze it.
            answer : 0
        example 2:    
            text : I apologize for the confusion. Unfortunately, I am unable to provide a specific answer without more context or clarification. If you can provide more information about the metrics you are referring to or specify the section of the document you are interested in, I would be happy to assist you further.
            answer : 0
        example 3:
            text : The document is titled "CodeGen2: LESSONS FOR TRAINING LLMS ON PROGRAMMING AND NATURAL LANGUAGES". It discusses the lessons learned from training Language Model (LM) systems on programming and natural languages. The document explores the performance of LM systems on various tasks, presents findings, and provides lessons and recommendations for training LM systems.
            answer : 1
        text : {answer}
        Return answer in the following format
            answer : 0 or 1
        
    '''

    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        max_tokens=MaxToken, 
        n=outputs
    )

    output = chat_completion.choices[0].message.content

    return output


def get_compare_summaries(question, text, MaxToken=400, outputs=3): 

    PROMPT = f'''
        your are a Research Assistant. 
        You are given a question and a text, you need to say whether the text is related answer to the question.
        If the text is related answer to the question give 1, if not give 0.

        for example, if the question is related to a graph/image/table then the text must be a summary of a graph/image/table.
        Or if the question is related to document then the text must be summary of a document not any other summaries.
        
 
        Question : {question}
        text : {text}

        Return answer in the following format
            answer : 0 or 1
        
    '''

    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        max_tokens=MaxToken, 
        n=outputs
    )

    output = chat_completion.choices[0].message.content

    return output


def get_sections(title, sections_list, MaxToken=4000, outputs=3): 

    PROMPT = f'''
        your are a Research Assistant. Your are given title of a research paper and list of headings. 
        you need to understand the title and use the list of headings given and 
        pick the possible sections in that paper from the given list only and return the answer in list format.
        You can keep the general sections that a Research paper could have like Introduction, Background, Observations, Limitations, Appendix, Abstract, Acknowledgement, References, Results, Conclusion etc...
        Do not use your own knowledge, answer must be from the given list.

        Title : {title}
        headings : {sections_list}
    '''

    client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "user", "content": PROMPT}
        ],
        max_tokens=MaxToken, 
        n=outputs
    )

    output = chat_completion.choices[0].message.content

    return output


def docx_to_pdf(file_path):

    pdf_file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
    pdf_file_name = pdf_file_name.replace('.','_') + '.pdf'

    outputFile =  f"./DocData/tempdir/{pdf_file_name}"
    #Create word document
    document = spire.doc.Document()
    document.LoadFromFile(file_path)
    #Save the document to a PDF file.
    document.SaveToFile(outputFile, spire.doc.FileFormat.PDF)
    document.Close()

    return file_path

def ppt_to_pdf(file_path):

    pdf_file_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
    pdf_file_name = pdf_file_name.replace('.','_') + '.pdf'

    outputFile =  f"./DocData/tempdir/{pdf_file_name}"
    print(outputFile)
    #Create a PPT document
    presentation = spire.presentation.Presentation()
    #Load PPT file from disk
    presentation.LoadFromFile(file_path)
    #Save the PPT to PDF file format
    presentation.SaveToFile(outputFile, spire.presentation.FileFormat.PDF)
    presentation.Dispose()

    return file_path

def convert_to_pdf(file_path):

    ext = file_path.split('.')[-1]

    if ext == 'docx':
        file = docx_to_pdf(file_path)
    elif ext in ['ppt','pptx','pptm']:
        file = ppt_to_pdf(file_path)
    
    return file


try:
    temp_path = str(sys._MEIPASS).replace('\\','/')
    poppler_path = temp_path+'/poppler-23.11.0/Library/bin'
    print('------> sys path\n',temp_path.split('/'))
except:
    cwd = os.getcwd()
    user_dir = '/'.join(cwd.split('\\')[:3])
    print('user dir -- > ', user_dir)
    poppler_path=f'{user_dir}/AppData/Local/anaconda3/envs/Doc2/Lib/site-packages/poppler-23.11.0/Library/bin'


def get_page_images_from_pdf(path, poppler_path=poppler_path):
    images = convert_from_path(path,
                                poppler_path=poppler_path)
    return images

def get_page_images_from_docx(file_path):

    image_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
    image_name = image_name.replace('.','_') + '.png'

    outputFile =  f"./DocData/pdf_1sPage_images/{image_name}.png"
    #Create word document
    document = Document()
    document.LoadFromFile(file_path)

    #Obtain image data in the default format of png,you can use it to convert other image format.
    imageStream = document.SaveImageToStreams(0, ImageType.Bitmap)

    with open(outputFile,'wb') as imageFile:
        imageFile.write(imageStream.ToArray())
    document.Close()

    return outputFile


def gptQA(file_path, sections_list = []):


    folder_path = './DocData/pdf_1sPage_images'
    os.makedirs(folder_path, exist_ok=True)
    
    titleSections_json = './DocData/PDF_titleSections.json'

    if os.path.isfile(titleSections_json):
        # if the json file exists
        with open(titleSections_json, 'r') as fp:
            json_data = json.load(fp)
    else:
        json_data = {}

    ts_dict = {'title':'',
               'sections':''}
    json_data[file_path] = ts_dict if file_path not in json_data.keys() else json_data[file_path]
    
    if file_path != None and json_data[file_path]['title'] == '':
        pdf_images = get_page_images_from_pdf(file_path)
        image_name = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        image_name = image_name.replace('.','_') + '.jpg'

        impath = folder_path + '/' + image_name
        pdf_images[0].save(impath, 'JPEG')
        # with open(impath, 'wb') as fp:
        #     fp.write(pdf_images[0])

        title = get_image_summary(impath)
        title = title if type(title) != tuple else title[-1]

        ts_dict['title'] = title 

        # title_update = title if len(title) > 1 else json_data[file_path]['title']
        json_data[file_path].update({'title':title})

    else:
        title = json_data[file_path]['title']
    
    if sections_list and len(json_data[file_path]['sections']) == 0 :
        sections = get_sections(title, sections_list) if len(sections_list) > 10 else sections_list
        ts_dict['sections'] = sections

        json_data[file_path].update({'sections':sections})
    else:
        sections = json_data[file_path]['sections']


    with open(titleSections_json, 'w') as fp:
        json.dump(json_data, fp) 


    return title, sections










sections_list = ['(a) Eliminating Rounding and Underflow error in INT4','(b) Breakdown of ouput errors in INT4','1 Introduction','2  for Xint = c1','2  for Xint = c2',
                 '2 Background','2.2 Weight and Activation PTQ for LLMs','2.3 Underflow for Reduced-Precision LLMs','4 Overcoming PTQ Underflow for LLMs','4.1 Observations',
                 '4.2 Integer with Denormal Representation','4.3 Advantages','5.2 Evaluation on Language Modeling Task','5.3 Evaluation on Zero-shot Reasoning Tasks',
                 '5.4 Evaluation on In-Context Learning Tasks','7 Limitation','A Appendix','A.2 Language modeling in >60B Models','A.4 Finding Scales for AQAS',
                 'A.5 Sweep of the Special Value in dINT','Abstract','Acknowledgement','Activation Quantization','E[(WX − (W +∆u +∆r)X)2] (4)',
                 'FP16 baseline 31.95 16.41 14.32 12.29 11.50','Figure 2: Absolute max value of (a) input activation and','Fused LN in OPT','Hi = ∂2E ∂W2','INT3',
                 'Layer Number Layer Number','Precision Format OPTQ','References',
]

sections_list = [' Models','1. General Purpose:','1. Multi-task:','10. Cross-Lingual Understanding:','11. Truthfulness:','12. Biases and Ethics in AI:','2. Coding:',
                 '2. Language Understanding:','3. Scientific Knowledge:','3. Story Cloze and Sentence Completion:','4. Dialog:','4. Physical Knowledge and World Understanding:',
                 '5. Challenges','5. Contextual Language Understanding:','5. Finance:','6. Commonsense Reasoning:','7. Reading Comprehension:','8. Mathematical Reasoning:',
                 '9. Problem Solving and Logical Reasoning:','Abstract—','Benchmark','Dataset Type Size/Samples Tasks Source Creation Comments','Goat','I. INTRODUCTION',
                 'II. BACKGROUND','III. LARGE LANGUAGE MODELS','IV. FINDINGS & INSIGHTS','IX. CONCLUSION','Index Terms—','REFERENCES','Sparrow','V. MODEL CONFIGURATIONS',
                 'VI. DATASETS AND EVALUATION','VII. SUMMARY AND DISCUSSION','VIII. CHALLENGES AND FUTURE DIRECTIONS',
]

# x,y = gptQA('./pdfdata/LLM Efficiency.pdf', sections_list)

# print('\n title : \n',  x)
# print('\n section: \n', y)




    # response = openai.Completion.create(  
    #     model="gpt-3.5-turbo-1106", 
    #     prompt=PROMPT, 
    #     max_tokens=MaxToken, 
    #     n=outputs 
    # ) 

    # output = list() 
    # for k in response['choices']: 
    #     output.append(k['text'].strip()) 

   