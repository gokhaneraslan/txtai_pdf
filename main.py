from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
## 9. RAG (Retrieval-Augmented Generation) ##
from txtai.pipeline import Extractor
from txtai import Embeddings


DATA_PATH = "data"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

documents = load_documents()
chunks = split_documents(documents)[0].page_content

data = [
    chunks
]


llm_embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2", content=True, autoid="uuid5")
llm_embeddings.index(data)

extractor = Extractor(llm_embeddings, "google/flan-t5-small")

llm_query = "What does gokhan doing?"
context = lambda question: [{"query": question, "question": f"Answer the following question using the context below.\nQuestion: {question}\nContext:"}]
print("RAG Result:")
print(extractor(context(llm_query))[0])
