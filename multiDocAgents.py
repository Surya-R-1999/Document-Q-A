
####################################################################


import openai
from configparser import ConfigParser
import os, sys

# from llama_index.readers.schema.base import Document
from llama_index import Document

from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
)
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata


from llama_index.agent import OpenAIAgent
from llama_index import load_index_from_storage, StorageContext
from llama_index.node_parser import SentenceSplitter

from llama_index.objects import ObjectIndex, SimpleToolNodeMapping

from llama_index.agent import FnRetrieverOpenAIAgent

import gptQA
from joblib import Parallel,delayed
from functools import partial
from llama_index import PromptTemplate
from llama_index.llms import OpenAI

from llmsherpa.readers import LayoutPDFReader

# ############################
### Load The openai keys
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



############################
### Use LLM Sherpa to extract section wise text
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"


#### LayoutPDFReader
def get_sherpa_docs(file_path, llmsherpa_api_url = llmsherpa_api_url):
    """
    extracts section wise text data from pdf.
    returns documents in llama-index document format
    inputs
        pdf_url : pdf url or path
    return
        sections : list of sections that are available in pdf
        docs : list of documents, created from pdf text
    """
    ## give relative path instead of absolute path

    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc_obj = pdf_reader.read_pdf(file_path)

    doc_title, _ = gptQA.gptQA(file_path = file_path, sections_list = [])
    sections = []
    docs = []
    for sec in doc_obj.sections():
        for chnk in sec.chunks():
            text = chnk.to_context_text()
            section_name = sec.title
            docs.append(Document(text=text, metadata = {'section':section_name, 
                                                        'file_path':file_path,
                                                        'title':doc_title}))
            
            sections.append(section_name)
    
    sections = [i.replace(',',' ') for i in set(sections)]
    _, sections = gptQA.gptQA(file_path = file_path, sections_list = sections)

    return sections, docs, doc_title


def get_summary_docs(file_path):
    docs = SimpleDirectoryReader(
        input_files=[file_path]
        ).load_data()
    return docs



#############################
#### Llama-Index Agents
## MultiDoc Agents

def create_Doc_agents(file, dir_path = './DocData/tempdir'):

    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)
    node_parser = SentenceSplitter()

    print(file)
    file_path = dir_path+'/'+file
    sections, docs, doc_title = get_sherpa_docs(file_path)
    print(doc_title,'\n')

    nodes = node_parser.get_nodes_from_documents(docs)

    vdb_path = f"./data/{dir_path+'/'+file.replace('.','_')}"
    if not os.path.exists(vdb_path):
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(
            persist_dir=vdb_path
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vdb_path),
        )

    # build summary index
    sum_docs = get_summary_docs(dir_path+'/'+file)
    sum_nodes = node_parser.get_nodes_from_documents(sum_docs)
    summary_index = SummaryIndex(sum_nodes, service_context=service_context)

    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    f"Useful for questions related to specific aspects of the document  {doc_title}."
                    f"The document contains the following sections  {sections}"
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=( #holistic
                    "Useful for any requests that require a  summary of the document  "
                    + doc_title + " or any of its sections. For questions about "
                    " more specific sections, please use the vector_tool."
                ),
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4", temperature=0.2)
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=False,
        system_prompt=f"""\
    You are a specialized agent designed to answer queries about a given document - title : {doc_title}.
    Allways generate a point-wise elaborated answers for given question.
    You must ALWAYS use at least one of the tools provided when answering a question.
    You must ALWAYS give some answer using one of the tools. if you are unable to answer using one tool use another one;
    do NOT rely on prior knowledge.\
    """,
    )

    ### Build Retriever-Enabled OpenAI Agent
    ## top-level agent to orchestrate across the different doc agents
    # define tool for each document agent
    wiki_summary = (
        f"This is a research paper. {doc_title}. Use"
        f" this tool if you want to answer any questions about the Document {doc_title} or any of its sections.\n"
        f"{sections}"
    )

    doc_tool = QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name=f"{file[:60].replace(' ','_').replace('.','_')}",
            description=wiki_summary,
        ),
    )

    return doc_tool


def create_top_agent(dir_path):
    
    # doc_ppt_files = [f'{dir_path}/{file}' for file in os.listdir(dir_path) if file.split('.')[-1] in ['ppt','pptx','pptm','docx']]
    # print(doc_ppt_files)
    # #covert the doc and ppt to pdf
    # converted_files = Parallel(n_jobs=8,prefer="threads")( delayed( partial(gptQA.convert_to_pdf) ) (file) for file in doc_ppt_files)
    
    # pdf files only
    pdf_files = [file for file in os.listdir(dir_path) if file[-4:] == '.pdf']

    all_tools = Parallel(n_jobs=8,prefer="threads")( delayed( partial(create_Doc_agents) ) (file) for file in pdf_files)

    # define an "object" index and retriever over these tools
    tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)

    obj_index = ObjectIndex.from_objects(
                    all_tools,
                    tool_mapping,
                    VectorStoreIndex,
                )

    top_agent = FnRetrieverOpenAIAgent.from_retriever(
        obj_index.as_retriever(similarity_top_k=15),
        system_prompt=""" \
            You are an agent designed to answer queries about given documents.
            Please always use the tools provided to answer a question. 
            Each tool will be giving outputs from different documents.
            Generate point-wise elaborated answers .  
            If you are unable to provide answer then just give the output of the tools.
            Do not rely on prior knowledge.\

            """,#for each tool's output separately
        verbose=False,
        )

    return top_agent


##########################
#### Multi query generator
def generate_queries1(query_str: str, llm = OpenAI(model="gpt-3.5-turbo"),  num_queries: int = 2, titles = []):
    """
    generates multiple queries related to user query
    input:
        query_str : query string
        llm : (optional) language model 
        num_quries : (optional) # queries to generate - default 2 queries
        titles : (optional) titles of the documents used
    return:
        list of generated queries
    """
    
    query_gen_prompt_str = (
        "You are a helpful assistant that generates multiple search queries based on a single input query."
        "the following are the document titles:"
        "{titles}"
        '''if the query is very general, 
            for example 
            what are the common points in the given doucments? or Are these papers related? 
            then generate questions that can retrieve specific sections of the paper and can be compared and summarized like
            Give the summary of the common sections in the given documents'''
        " Generate {num_queries} search queries, one on each line, "
        "related to the documents using the following input query:\n"
        "Query: {query}\n"
        "Queries:\n"
        )
    query_gen_prompt = PromptTemplate(query_gen_prompt_str)


    fmt_prompt = query_gen_prompt.format(
        num_queries=num_queries, 
        query=query_str, 
        titles=titles
    )

    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")[:2]

    print('generated Questions ==========')
    print(queries)
    print('===============================')

    return queries  # ' \n '.join(queries) 

###########################


# def qery_top_agent(query):
#     res = top_agent.query(query)
#     return res




# q = 'summary'
# res_top = top_agent.query(q)
# res_base = base_query_engine.query(queries)

# print('\ntop-agent\n',str(res_top))
# print('\nbase\n',str(res_base))

# agents = {}
# query_engines = {}

# this is for the baseline
# all_nodes = []

# all_tools = []