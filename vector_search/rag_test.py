from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("my_notes.pdf")
pages = loader.load()
#for i, doc in enumerate(pages):
#    print(f"Chunk {i}: {doc.page_content[:100]}")  # print first 100 chars
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)
oakey = "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"
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
embeddings = OpenAIEmbeddings(openai_api_key=oakey)

from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)


from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=oakey),
    retriever=db.as_retriever()
)

result = qa_chain.invoke({"query": "What are hyperbolic functions?"})
print(result)