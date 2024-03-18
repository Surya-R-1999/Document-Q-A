import os, sys
import streamlit as st
from langchain.agents import tool
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import (ConversationalRetrievalChain,RetrievalQA)
from llama_index import SimpleDirectoryReader
# from IPython.display import Markdown, display
# from llama_index.text_splitter import SentenceSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.llms import OpenAI
# from langchain.llms import OpenAIChat

from langchain_community.chat_models import ChatOpenAI
# from langchain_community.llms import OpenAI

# from langchain.llms import Cohere
from langchain_community.llms import Cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from langchain.agents import OpenAIFunctionsAgent
from langchain.schema import SystemMessage
from langchain.agents import AgentExecutor

from PIL import Image
from streamlit_option_menu import option_menu
import pandas as pd
from metaphor_python import Metaphor
import extract_msg
import shutil
from langchain.docstore.document import Document
import json

import DocImageProcess
import table_extraction
import multiDocAgents
import gptQA
import win32com.client
import pythoncom
# import ppt2pptx
from joblib import Parallel,delayed
from functools import partial

import os


# ## OPEN AI KEY
# os.environ["OPENAI_API_KEY"] =  "sk-exuLee6vEslOwSIrbglKT3BlbkFJpFqetblIBMz1ZV70nzBh000"
# open_ai_key =  "sk-exuLee6vEslOwSIrbglKT3BlbkFJpFqetblIBMz1ZV70nzBh000"

# Coherekey= "mvjcvGo5aLA4YBqX9JSdOrQy3vVyPVQAmmQzDcdR000"
# os.environ["Coherekey"] = Coherekey
# os.environ["METAPHOR_API_KEY"] = "9dc52932-9821-419e-9910-59af6d8c5e96000"
# client = Metaphor(api_key=os.environ["METAPHOR_API_KEY"])

## LOGO
# logo = Image.open("logo.png")
try:
    temp_path = str(sys._MEIPASS).replace('\\','/')
    logo = Image.open(temp_path+'/DocQA/logo.png')
    print('------> sys path\n',temp_path.split('/'))
except:
    logo = Image.open("logo.png")


## PROMPT TEMPLATE
#################################################

prompt_template = """Use the following pieces of context to answer the question at the end. If the answer can be inferred from the context, provide it with proper punctuation and capitalization. 
If the answer is not present in the context but can be addressed with your model knowledge up to your last update, use that information to answer. 
If you still don't know the answer, state that you don't know. Do not fabricate an answer.

{context}

Question: {question}
Generate point-wise answers for the question if required, and format them clearly with bullet points or numbers.
Conclude or explain the context based on the question's intent when possible.
If the answer is not present in the context, attempt to provide a close enough answer using the model's knowledge without making assumptions beyond that knowledge.
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]#, "sources"]
)
question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question.
Return any relevant text, if none of the text is relevant then return the summary of the whole text. 
{context}
Question: {question}
Relevant text, if any:"""
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template, input_variables=["context", "question"]
)

combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

QUESTION: {question}
=========
{summaries}
=========
Answer:"""
COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
)

multi_q_DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is 
    to generate five different versions of the given user 
    question to retrieve relevant documents from a vector  database.
    There may be some generic questions like 'give summary of the file', 'provide summary of the document' etc... 
    for those questions generate  
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}""",
)


## BACKEND LOGIC
#################################################

# 1) Loading Documents
def load_docs(filepath):
    reader = SimpleDirectoryReader(input_dir=filepath, exclude=["*.png","*.jpg","*.jpeg", '*.tiff','*.tif', '*.pdf'])
    documents = reader.load_langchain_documents()
    # docs=[d.to_langchain_format() for d in documents]
    return documents

# 2) Splitting the Documents
def split_docs(documents, chunk_size=5000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                    chunk_overlap=chunk_overlap,
                                                    separators = ['\n','\n\n','.','\t'])
    docs = text_splitter.split_documents(documents)
    print("=================>>>>>> Chunking")
    return docs

# 3) Create Vectors for Chunks
def get_basevectorDB(chunks):
    
    try:
        st.session_state["base_retriever"].delete_collection()
        print("\nDeleted Previous Collection/VectorDB, Creating new one-----")
    except Exception as e:
        print('Error in deleting the DB----> \n',e)
        print("\nCreating the First Vector DB ------------------------------")
    if len(chunks) > 0:
        vectordb = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())#, persist_directory="./DocData/chroma_db")
    else:
        chunks = [Document(page_content='', metadata={'text_len':0})] # if a file doesn't contain text (only images, graphs)
        vectordb = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    # db3 = Chroma(persist_directory="./DocData/chroma_db", embedding_function=OpenAIEmbeddings())

    return vectordb

# 4) MutliQueryRetriever

# Set logging for the queries

def get_multQretriever(vectordb):
    retriever_from_llm = MultiQueryRetriever.from_llm(
                            retriever=vectordb.as_retriever(
                                search_type='mmr', 
                                search_kwargs={'k': 4,                                
                                            'lambda_mult': 0.2,
                                            'fetch_k': 20,
                                            # 'filter':{'source':filePath}, # only from which pdf file it should fetch the relavent chuncks
                                            "score_threshold":0.8,
                                            }
                            ),prompt=multi_q_DEFAULT_QUERY_PROMPT,
                            llm=ChatOpenAI(temperature=0.2)
    )
    return retriever_from_llm

# 5) Complete Creation of Database from Loading the Document to Creation of Vector Database
def create_database(path):
    # with st.spinner('Please Wait the Document is under process ...'):
    try:
        print("Loading the documents ------>")
        documents=load_docs(path)
        print("Document Loaded")
    except Exception as e:
        print("Error in loading the documents")
        print(e)
        documents = Document(page_content='', metadata={})
        print(type(documents), '\n',documents)

    try:
        chunks=split_docs(documents)
    except Exception as e:
        print("Error in splitting the documents")
        print(e)
        chunks = []

    try:
        vectordb=get_basevectorDB(chunks)
    except Exception as e:
        print("Error in creating the vector database")
        print(e)
            
    # st.success('Done!')
    return vectordb

# 6) Function to Get Answers of Query from the Vector Database
def get_relevant_answer(query,db):
    embeddings_filter=EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.70)
    retriever=ContextualCompressionRetriever(base_compressor=embeddings_filter,base_retriever= get_multQretriever(db))
    try:
        compressed_docs= retriever.get_relevant_documents(query)
        try:
            if len(compressed_docs)>4:
                compressed_docs=compressed_docs[:3]

            elif len(compressed_docs)==0:
                raise Exception("No relevant documents found")
            else:
                pass
        finally:
            try:
                # llm_model = OpenAIChat(temperature=0.25,model_name="gpt-3.5-turbo-1106")
                llm_model = ChatOpenAI(temperature=0.25,model_name="gpt-3.5-turbo-1106")
                chain=load_qa_chain(llm_model,  chain_type="stuff", prompt=PROMPT)
                res=chain({"input_documents": compressed_docs, 
                                        "question":[ query]})
                print('--- Using OpenAi - gpt-3.5-turbo-1106 Model ---')
            except Exception as e:
                print('ignore ---',e)
                llm_model = Cohere(cohere_api_key=Coherekey)
                chain=load_qa_chain(llm_model,  chain_type="stuff", prompt=PROMPT)
                # map_red_chain = load_qa_chain(OpenAI(temperature=0.2),  chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
                res=chain({"input_documents": compressed_docs, 
                                        "question":[ query]})
                print('--- Using Cohere Model ---')
            
            return res, compressed_docs
    except Exception as e:
        print(e)
        return "No relevant documents found", compressed_docs

# 7) Logic For Metaphor API Tools
@tool
def search(query: str):
    """Call search engine with a query."""
    return client.search(query, use_autoprompt=True, num_results=5)

@tool
def get_contents(ids: List[str]):
    """Get contents of a webpage.
    The ids passed in should be a list of ids as fetched from `search`.
    """
    return client.get_contents(ids)
@tool
def find_similar(url: str):
    """Get search results similar to a given URL.
    The url passed in should be a URL returned from `search`
    """
    return client.find_similar(url, num_results=5)
tools = [search, get_contents, find_similar]


# 8) Function to Retrieve Answers from Internet Using Metaphor API Tools
def get_relevant_answer_internet(query):
    llm = ChatOpenAI(temperature=0.2, max_retries=2)
    prompt = OpenAIFunctionsAgent.create_prompt(system_message=SystemMessage(
        content="You are a web researcher who uses search engines to look up information."
    ))
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False) # True
    result = agent_executor.run(query)
    return result

# 9) Function to Download attachments present in outlook mail
def load_email(email_path):
    msg = extract_msg.openMsg(email_path)
    save_path=os.path.join('./DocData/tempdir/')
    for f in msg.attachments:
        if f.name.split('.')[-1] == 'msg':
            print('customfilename ', f.name)
            filename = str(f.name).replace(':',"")
            f.save(customPath=save_path, customFilename=filename, extractEmbedded=True)
        else:
            name = ' '.join(str(f.longFilename).split('.')[:-1])
            f_type = str(f.longFilename).split('.')[-1]
            date = msg.date.strftime("%m-%d-%Y").replace('/', '-')
            filename = f"{name}_{date}.{f_type}"
            # print('customfilename 2 - > ', filename, '\nFtype  ', f_type)
            filename = filename.replace(':',"")
            f.save(customPath=save_path, customFilename=filename)
    print("Loaded the attachments in the mail....")

def display_logo_and_title():
    primary_color = '#027dc3'
    header = st.container()
    # header.markdown("Explore Your Knowledge Base")
    header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)
    with header:
        col1, col3 = st.columns([30,6.5])
        with col1:
            st.markdown(f'<h1 style="color:{primary_color};">S.A.G.E.</h1>', unsafe_allow_html=True)
        with col3:
            st.image(logo, width=240)
    # header.title("Here is a sticky header")
    # header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

    ### Custom CSS for the sticky header
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.5rem;
            background-color: white;
            z-index: 999;
            color: #027dc3;
        }
    </style>
        """,
        unsafe_allow_html=True
    )
    
    
def set_page_container_style():
    padding_top = 1.6
    st.markdown(f"""
        <style>
        .block-container{{
            padding-top: {padding_top}rem;    }}
    </style>""",
        unsafe_allow_html=True,
    )


def ppt_to_pptx(inppt):
    """
    Funtion that converts ppt file to pptx file - to make it a zip file
        pptx file will be saved in the same directory as inppt file.

    inputs:
        inppt : input ppt file path - provide full absolute path
        
    """
    PptApp = win32com.client.Dispatch("Powerpoint.Application", pythoncom.CoInitialize())
    PptApp.Visible = True
    PPtPresentation = PptApp.Presentations.Open(inppt)
    PPtPresentation.SaveAs(inppt+'x', 24)
    PPtPresentation.close()
    PptApp.Quit()



## SESSION STATE 
#########################################
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state["llm"] = ChatOpenAI(temperature=0.25)

if "retriever_from_llm" not in st.session_state:
    st.session_state["retriever_from_llm"] = False

if "questin_lot" not in st.session_state:
    st.session_state["questin_lot"] = []

if "assistant_message" not in st.session_state:
    st.session_state.assistant_message = ''

if "chat_history" not in st.session_state:
    st.session_state.chat_history = pd.DataFrame()

if 'ctr' not in st.session_state:
    st.session_state['ctr'] = 0

if "files_list" not in st.session_state:
    st.session_state.files_list = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0


## FRONTEND
################################################

## BASIC TITLE AND MARKDOWN


st.set_page_config(layout="wide")

set_page_container_style()
display_logo_and_title()

## 1) SIDEBAR WITH FILE UPLOAD FUNCTIONALITY
with st.sidebar:
    option = option_menu(
        menu_title = None,
        options = ["Chat","Chat History"],
        orientation="vertical",
        icons=['chat','list-task'],
    )

    uploadedFiles = st.file_uploader(" ", accept_multiple_files=True) 
    files_uploaded = [file.name for file in uploadedFiles]

    if uploadedFiles and option != 'Chat History' and (len(st.session_state.files_list) == 0 or (st.session_state.files_list != files_uploaded)):
        print(" -- FILES UPLOADED -- ")
        print('Number of files uploaded ---> ',len(uploadedFiles))
        st.session_state.files_list = files_uploaded.copy()
        tempfilDir = './DocData/tempdir'

        try:
            # remove previous files 
            os.makedirs('./DocData/tempdir', exist_ok=True)
            # Iterate over each entry in the directory & remove previous files/directories
            for entry in os.listdir('./DocData/tempdir'):
                full_path = os.path.join('./DocData/tempdir', entry)
                # If entry is a file, delete it
                if os.path.isfile(full_path):
                    os.remove(full_path)
                    print(f"File {full_path} has been deleted.")
                # If entry is a directory, delete it and its contents
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                    print(f"Directory {full_path} has been deleted along with its contents.")

            os.makedirs(tempfilDir, exist_ok=True)

            with st.spinner('Please Wait the Document is under process ...'):  
                for file in uploadedFiles:
                    print(file.name)
                    filepath = tempfilDir + f'/{file.name}'
                    file_ext = str(file.name).split('.')[-1]
                    bytedata = file.getvalue()
                    filename = ''.join(str(file.name).split('.')[:-1])
                    
                    if file_ext == 'msg':
                        # filepath = './DocData/emailData' + f'/{file.name}'
                        with open(filepath, 'wb') as fp:
                            fp.write(bytedata)
                        load_email(filepath)
                        # Iterate over each entry in the directory & move to cropped images folder if format found to be jpg,jpeg,png
                        
                        print(filename)

                        dest_dir = f"./DocData/cropped_images/{filename}/"
                        os.makedirs(f'./DocData/cropped_images/{filename}/')
                        for file in os.listdir('./DocData/tempdir'):
                            full_path = os.path.join('./DocData/tempdir', file)
                            # If entry is a iamge file, move it
                            if os.path.isfile(full_path) and str(full_path).split(".")[-1] in ['jpg','jpeg','png']:
                                # Define the destination path for the file
                                dest_path = os.path.join(dest_dir, file)

                                # Move the file to the destination directory
                                shutil.move(full_path, dest_path)
                                print(f"File {full_path} has been moved to {dest_path}.")
                    else:
                        # bytedata = file.getvalue()
                        with open(filepath, 'wb') as fp:
                            fp.write(bytedata)

                        if file_ext == 'ppt':
                            print(f'./DocData/tempdir/{file.name}')
                            pwd = os.getcwd()
                            ppt_to_pptx(pwd + r".\\DocData\\tempdir\\"+file.name)
                            os.remove(f'./DocData/tempdir/{file.name}')

                print("----File Saved--------")

                ##############
                # covert the doc and ppt to pdf
                doc_ppt_files = [f'{tempfilDir}/{file}' for file in os.listdir(tempfilDir) if file.split('.')[-1] in ['ppt','pptx','pptm','docx']]
                print(doc_ppt_files)
                
                # converted_files = Parallel(n_jobs=8,prefer="threads")( delayed( partial(gptQA.convert_to_pdf) ) (file) for file in doc_ppt_files)
                for file in doc_ppt_files:
                    _ = gptQA.convert_to_pdf(file)
                    print(_)

                #delete docx, ppt files
                for file in doc_ppt_files:
                    os.remove(file)
                ##############

                available_docs = os.listdir(tempfilDir)
                print("No of available docs ---> ", len(available_docs))
                print(available_docs)

                st.session_state["base_retriever"] = create_database(tempfilDir)

                st.session_state['file_formats'] = set([i.split('.')[-1] for i in available_docs])
                print(st.session_state['file_formats'])



                ############## image/figure/graph processing
                doc_list_img = []

                for doc in available_docs:
                    doc_path = f'{tempfilDir}/{doc}'

                    try:
                        summary_json_path = DocImageProcess.image_process_pipeline(doc_path, api_key=os.environ["OPENAI_API_KEY"])

                        with open(summary_json_path, 'r') as fp:
                            json_data = json.load(fp)

                        for key in json_data.keys():
                            doc =  Document(page_content=str(json_data[key]), 
                                            metadata={
                                                "image_path": json_data[key]["image_path"], 
                                                "file_name":key})
                            doc_list_img.append(doc)
                    except FileNotFoundError as e:
                        print(doc, '-- has no images/graphs/figures')
                    except Exception as e:
                        print("Error in DocImageProcess for Document -- ", doc)
                        print(e)

                # print('example doc\n',doc_list_img[0])
                
                
                ################## table extraction and processing - pdf
                summary_json_list = table_extraction.main(tempfilDir)
                
                doc_list_tbl = []
                for jfile in summary_json_list:
                    jfile_path = './DocData/SummaryJson/'+jfile
                    
                    with open(jfile_path, 'r') as fp:
                        json_data = json.load(fp)
                    
                    for key in json_data.keys():
                        if json_data[key] != None:
                            doc = Document(page_content = str(json_data[key]),
                                        metadata = {
                                            'image_path' : json_data[key]['image_path'],
                                            'file_name' : key
                                            }) 
                            doc_list_tbl.append(doc)
                    

                if len(doc_list_img) > 0 : st.session_state['base_retriever'].add_documents(doc_list_img)
                if len(doc_list_tbl) > 0 : st.session_state['base_retriever'].add_documents(doc_list_tbl)

                st.session_state['multidocAgent'] = multiDocAgents.create_top_agent(tempfilDir)
                print("base retriever -- > Done")
            st.success('Done!') 

        except Exception as e :
            print("Error in processing--> \n",e)




## 2) CHAT
if option == 'Chat':
    st.markdown("<h3 style='color:#A9A9A9;'>Chat Session</h3>", unsafe_allow_html=True)
    if st.session_state.query_count != 0 or len(st.session_state.files_list) != 0:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    else:
        if st.session_state.messages:
            st.session_state.messages.pop() 

    if prompt := st.chat_input(""):
        print("=============================\nuser Q ---> ", prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"): # to get typing/ans generating effect - use empty & with  
            message_placeholder = st.empty()
            full_response = ""
            filename = ''
            page_id = "-"
            source_set = set()
            using_Internet = False
            empty_response = False

            try:

                multiDocAgent_Res = st.session_state['multidocAgent'].query(prompt)
                
                res_class = gptQA.get_class(str(multiDocAgent_Res))
                print('-->', res_class, '<--')
                res_class = res_class.split(':')[-1]
                res_class = int(res_class) if res_class.strip().isnumeric() else 0
                if res_class or len(str(multiDocAgent_Res)) >= 300:
                    print(str(multiDocAgent_Res))
                    st.write('### Engine 1')
                    st.write(str(multiDocAgent_Res))

                if len(st.session_state.files_list) != 0: # and not int(res_class):
                    try:
                        db =  st.session_state["base_retriever"]
                    except:
                        print('base retriever not done....')
                    full_response, compressed_docs = get_relevant_answer(prompt, db)

                    # print(" --- base retreiver response --- ", full_response['output_text'])
                    # print(compressed_docs)

                    if len(compressed_docs) > 0 and full_response['output_text'] != "I don't know." :  
                        print('No of docs fetched and compressed/filtered ---> ',len(compressed_docs))
                        # print(res['input_documents'][0].to_json()["kwargs"]['metadata'])
                        with st.sidebar:
                            st.write('**Sources**')
                            images_list = set()
                            for ans in full_response["input_documents"]:
                                # print("PRINTING ANS JSON =============================")
                                # print(ans.metadata) 
                                metadata = ans.metadata #ans.to_json()['kwargs']['metadata']
                                # print(metadata)
                                try:
                                    print(metadata)
                                    filename = metadata['file_name'].split("\\")[-1]
                                except KeyError:
                                    filename = metadata['file_path'].split("\\")[-1] if 'file_path' in metadata.keys() else ''
                                except Exception as e:
                                    print(e)
                                    
                                if 'image_path' in metadata.keys():
                                    related = gptQA.get_compare_summaries(prompt, ans.page_content)
                                    print(related)
                                    related = related.split(':')[-1]
                                    if related:
                                        images_list.add(metadata['image_path'])

                                if filename.split(".")[-1] in ["ppt","pptx","pptm"]:
                                    page_id = "-"
                                else:
                                    try:
                                        page_id = ans.to_json()['kwargs']['metadata']['page_label']
                                    except:
                                        pass

                                source_set.add(filename)
                                st.write(filename + "  , page : " + str(page_id))
                        print("\n\n============================ Result ===============================")

                        print(full_response['output_text'])
                        st.write('### Engine 2')
                        st.write(full_response['output_text'])
                        st.session_state["assistant_message"] = full_response['output_text']
                        

                        for img in images_list:
                            image = Image.open(img)
                            st.image(image, width=800)

                    else:
                        print("No data Found via Documents")
                        empty_response = True
                if len(st.session_state.files_list) == 0 or empty_response:
                    print("Redirecting to use Internet Knowledge...")
                    try:
                        full_response = get_relevant_answer_internet(prompt)

                        if full_response:
                            with st.sidebar:
                                st.write("**Using Internet Knowledge**")

                            st.session_state["assistant_message"] = full_response
                            st.write(full_response)
                            using_Internet = True
                            st.session_state.query_count = 1
                        else:
                            print("No data Found on Internet")
                            st.write("Please rephrase the question and try again.")
                            st.session_state["assistant_message"] = 'No Data found on Internet..'
                        
                    except:
                        st.session_state.messages = []
                        print("Error with API to retrieve knowledge from Internet..")
                        st.write("Please rephrase the question and try again.")
            finally:
                
                if not os.path.exists("Qlog.txt"):
                    with open("Qlog.txt",'w') as fp:
                        fp.write("filename;page_id;prompt;assistant_message\n")
                        print("File Created")
                        fp.close()

                with open('Qlog.txt', 'a+') as fp:
                    print("Adding Elements to File")
                    if using_Internet:
                        try:
                            fp.write("\n" + "Internet" + ";" "-" + ";" + prompt + ";" + st.session_state["assistant_message"].replace("\n", " "))
                        except:
                            pass
                        fp.close()
                    elif len(filename) > 0:
                        try:
                            fp.write("\n" + f"{filename}" + ";" f"{page_id}" + ";" + prompt + ";" + st.session_state["assistant_message"].replace("\n", " "))
                        except:
                            pass
                        fp.close()


                st.session_state["assistant_message"] = str(multiDocAgent_Res) + '\n' + st.session_state["assistant_message"] # if res_class else st.session_state["assistant_message"]
                st.session_state.messages.append({"role": "assistant", "content": st.session_state["assistant_message"]})
            

if option == "Chat History":
    WELCOME_MESSAGE = """\

        View the Chat History of the ChatBot.
        """
    st.markdown(WELCOME_MESSAGE)
    st.markdown("<h2 style='color:#027dc3;font-weight:bold;'>Chat History</h2>", unsafe_allow_html=True)
    if os.path.exists("Qlog.txt"):
        st.session_state.chat_history = pd.read_csv('Qlog.txt',sep=';', on_bad_lines='skip')
    
    if st.session_state.chat_history.empty:
        st.write("No Chat History..")
    else:
        chat_log = st.session_state.chat_history
        chat_log.set_index('prompt', inplace=True)
        st.table(chat_log)