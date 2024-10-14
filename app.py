# !pip install firecrawl-py llama_index kdbai_client llama-index-vector-stores-kdbai
from firecrawl import FirecrawlApp
import kdbai_client as kdbai
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Firecrawl setup and crawling
app = FirecrawlApp(api_key='fc-bae517a8517d414a8969c0aeda3f611e')
crawl_result = app.crawl_url(
    'https://code.kx.com/kdbai',
    params={
        'crawlerOptions': {
            'limit': 10,
            'includes': ['kdbai/*']
        },
        'pageOptions': {
            'onlyMainContent': True
        }
    },
    wait_until_done=True
)

# KDB.AI setup
session = kdbai.Session(endpoint="https://cloud.kdb.ai/instance/dkdtpq6n42", api_key="2272da68ac-jWtJBh+HDZOIVT6o6BwK1MSPnilOhSNN8YGq1PBMgS/3HsdkeFhvv/hDFlgN2H3Xg6GobzC78DA42t5w")
# our schema includes extra metadata fields in case we want to filter by them
schema = {
    "columns": [
        {"name": "document_id", "pytype": "bytes"},
        {"name": "text", "pytype": "bytes"},
        {
            "name": "embedding",
            "vectorIndex": {
                "type": "flat",
                "metric": "L2",
                "dims": 1536
            }
        },
        {"name": "title", "pytype": "bytes"},
        {"name": "sourceURL", "pytype": "bytes"},
        {"name": "lastmod", "pytype": "datetime64[ns]"}
    ]
}
table = session.create_table("documentation", schema)

# Process and index documents
documents = [Document(text=item['content'], metadata=item['metadata']) for item in crawl_result]
vector_store = KDBAIVectorStore(table)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=OpenAIEmbedding()
)

# Create query engine
query_engine = index.as_query_engine()

# Example query
response = query_engine.query("What are the system requirements for KDB.AI?")
print(response)
