# Input data
# Your notes — could be a markdown file, a bunch of .txt files, or a single PDF.

# Chunking
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(your_raw_notes)


# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors = embedder.embed_documents(chunks)


# Vector Store
from langchain.vectorstores import Chroma
db = Chroma.from_texts(chunks, embedding=embedder)

# Retrieval
relevant_chunks = db.similarity_search("What is supervised learning?", k=4)


# Generation (so actual RAG)
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
response = qa_chain.run("What is supervised learning?")
