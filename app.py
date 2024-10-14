import streamlit as st
import os
from firecrawl import FirecrawlApp
import kdbai_client as kdbai
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.kdbai import KDBAIVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# Set up Streamlit page
st.set_page_config(page_title="Documentation Chatbot", page_icon="🤖")
st.title("Documentation Chatbot")

# Function to load or create index
@st.cache_resource
def load_or_create_index():
    # Check if the index already exists
    if 'index' not in st.session_state:
        # Firecrawl setup and crawling
        firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        app = FirecrawlApp(api_key=firecrawl_api_key)
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
        kdbai_endpoint = os.getenv('https://cloud.kdb.ai/instance/dkdtpq6n42')
        kdbai_api_key = os.getenv('cbe0f7f340-eQYoca1WSX398PJj8ehUPx3jvrA+uHNQEOCjZQ9OTEjNE5vBugGpLofoQeIQzJ8su504dLHdPazEwnOH')
        session = kdbai.Session(endpoint=kdbai_endpoint, api_key=kdbai_api_key)
        
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
        
        openai_api_key = os.getenv('OPENAI_API_KEY')
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=OpenAIEmbedding(api_key=openai_api_key)
        )
        
        st.session_state['index'] = index
    
    return st.session_state['index']

# Load or create the index
index = load_or_create_index()

# Create query engine
query_engine = index.as_query_engine()

# User input
user_query = st.text_input("Ask a question about the documentation:")

if user_query:
    # Get response from the chatbot
    response = query_engine.query(user_query)
    
    # Display the response
    st.write("Chatbot: ", response)

# Add some information about the chatbot
st.sidebar.title("About")
st.sidebar.info(
    "This chatbot uses Firecrawl to fetch documentation, "
    "stores it in KDB.AI, and uses LlamaIndex for querying. "
    "Ask any question about the KDB.AI documentation!"
)
