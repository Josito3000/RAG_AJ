from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
import os

class RAGEngine:
    def __init__(self, documents_dir="./data", model_name="all-MiniLM-L6-v2"):
        self.documents_dir = documents_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.db = None
        self.initialize()
        
    def initialize(self):
        """Carga documentos y construye la base de vectores"""
        # Carga documentos
        if not os.path.exists(self.documents_dir):
            os.makedirs(self.documents_dir)
            
        loader = DirectoryLoader(self.documents_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        if not documents:
            print("No se encontraron documentos para indexar")
            return
            
        # Divide documentos en chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Crea la base de vectores
        self.db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db",
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Base de conocimiento creada con {len(texts)} fragmentos")
        
    def query(self, question, k=3):
        """Consulta la base de conocimiento"""
        if not self.db:
            return "Base de conocimiento no inicializada"
            
        # Recupera documentos relevantes
        docs = self.db.similarity_search(question, k=k)
        
        # Formato simple de respuesta basada en contexto
        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Basado en los documentos, encontré esta información:\n\n{context}"