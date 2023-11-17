# Documentation for LlamaIndex:

**Semantic Search** : 

LlamaIndex provides a simple in-memory vector store for semantic search. The following code demonstrates how to use LlamaIndex for semantic search:
```
from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```
**Synthesis over Heterogeneous Data** :

LlamaIndex supports synthesizing across heterogeneous data sources by composing a graph over your existing data. Specifically, compose a summary index over your subindices. A summary index inherently combines information for each node, allowing it to synthesize information across your heterogeneous data sources. The following code demonstrates how to synthesize data from different sources:
```
from llama_index import VectorStoreIndex, SummaryIndex
from llama_index.indices.composability import ComposableGraph

index1 = VectorStoreIndex.from_documents(notion_docs)
index2 = VectorStoreIndex.from_documents(slack_docs)
graph=ComposableGraph.from_indices(SummaryIndex,[index1,index2], 
index_summaries=["summary1", "summary2"])
query_engine = graph.as_query_engine()
response = query_engine.query("<query_str>")
```
**Routing over Heterogeneous Data** : 

With LlamaIndex, you can route to the corresponding source based on the query provided. First, build the sub-indices over different data sources. Then construct the corresponding query engines, and give each query engine a description to obtain a QueryEngineTool. The following code demonstrates how to route queries to the appropriate source:
```
from llama_index import TreeIndex, VectorStoreIndex
from llama_index.tools import QueryEngineTool

...

# define sub-indices
index1 = VectorStoreIndex.from_documents(notion_docs)
index2 = VectorStoreIndex.from_documents(slack_docs)
index3 = VectorStoreIndex.from_documents(google_docs)

# define query engines and tools
tool1 = QueryEngineTool.from_defaults(
    query_engine=index1.as_query_engine(),
    description="Use this query engine to do...",
)
tool2 = QueryEngineTool.from_defaults(
    query_engine=index2.as_query_engine(),
    description="Use this query engine for something else...",
)
tool3 = QueryEngineTool.from_defaults(
    query_engine=index3.as_query_engine(),
    description="Use this query engine for a different purpose...",
)

# define a RouterQueryEngine over them
router = RouterQueryEngine.from_defaults(
    query_engine_tools=[tool1, tool2, tool3],
)

# use the router to route queries
response = router.query(
    "In Notion, give me a summary of the product roadmap."
)
print(response)

# use a different query engine for a different query
response = tool2.query_engine.query(
    "Show me all meetings scheduled for next week in Slack."
)
print(response)
```
In this example, we first define three sub-indices: index1, index2, and index3. Each sub-index is created by initializing a VectorStoreIndex with documents from different sources: notion_docs, slack_docs, and google_docs, respectively.

Next, we define three QueryEngineTool objects: tool1, tool2, and tool3. Each QueryEngineTool is associated with a particular sub-index, as defined by their query_engine attributes. The description attribute provides a way to differentiate between query engines based on their descriptions.

We then create a RouterQueryEngine that uses a LLMSingleSelector as the router. The LLMSingleSelector uses the LLM to choose the best sub-index to route the query to, based on the descriptions provided in the query engine tools.

Finally, we use the RouterQueryEngine to route queries. In the first query, we route a semantic search query to the index1 sub-index, which was associated with the query engine for "In Notion, give me a summary of the product roadmap." In the second query, we route a query about meetings in Slack to the index2 sub-index, which was associated with the query engine for "Show me all meetings scheduled for next week in Slack."

This way, you can route queries to the most suitable sub-index based on their descriptions, allowing you to synthesize information across your heterogeneous data sources more effectively.

**Summarization** : 

Summarization is the process of distilling the most important or relevant information from a large amount of text. In the context of LlamaIndex, summarization involves creating a summary index from a collection of documents.

To create a summary index, you can use the SummaryIndex class from the LlamaIndex library. The SummaryIndex class takes a list of documents as input and creates a summary index that reflects the important information in the documents. You can then use this summary index to perform queries on the summarized information.

Here is an example code snippet for creating a summary index:

```
from llama_index import SummaryIndex

# Load the documents
documents = [{"title": "Document 1", "content": "This is the content of document 1."},
              {"title": "Document 2", "content": "This is the content of document 2."}]

# Create a summary index
index = SummaryIndex.from_documents(documents)

# Use the summary index to perform a query
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic of this document?").to_dataframe()

print(response)

```
The to_dataframe() method is used to convert the response from an object to a pandas dataframe, which is then printed to the console.
