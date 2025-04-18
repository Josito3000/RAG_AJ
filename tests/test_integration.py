#Python built-in libraries
import os
import shutil
#1st party libraries
from app.rag_engine import RAGEngine
#3rd party libraries
import pytest

class TestRAGIntegration:
    @classmethod
    def setup_class(cls):
        """Prepara el entorno de prueba"""
        # Crea un directorio temporal para pruebas
        os.makedirs("./test_data", exist_ok=True)
        
        # Crea un documento de prueba
        with open("./test_data/sample.txt", "w") as f:
            f.write("Python es un lenguaje de programación interpretado. "
                   "Flask es un framework web minimalista para Python. "
                   "FastAPI es un framework moderno para construir APIs.")
    
    @classmethod
    def teardown_class(cls):
        """Limpia después de las pruebas"""
        # Elimina directorios temporales
        if os.path.exists("./test_data"):
            shutil.rmtree("./test_data")
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
    
    def test_document_ingestion(self):
        """Prueba que los documentos se carguen correctamente"""
        engine = RAGEngine(documents_dir="./test_data")
        assert engine.db is not None
    
    def test_query_retrieval(self):
        """Prueba que las consultas devuelvan resultados relevantes"""
        engine = RAGEngine(documents_dir="./test_data")
        
        # Consultas de prueba
        python_query = "¿Qué es Python?"
        framework_query = "¿Qué framework es minimalista?"
        
        # Verifica que las respuestas contengan información relevante
        python_response = engine.query(python_query)
        print("\nRespuesta a '¿Qué es Python?':")
        print(python_response)
        assert "Python" in python_response
        assert "lenguaje de programación" in python_response
        
        framework_response = engine.query(framework_query)
        print("\nRespuesta a '¿Qué framework es minimalista?':")
        print(framework_response)
        assert "Flask" in framework_response
        assert "minimalista" in framework_response
    
    def test_end_to_end_flow(self):
        """Prueba flujo completo de ingesta a consulta"""
        # Crear nuevo documento durante la prueba
        with open("./test_data/new_doc.txt", "w") as f:
            f.write("PyTorch es una biblioteca de aprendizaje profundo para Python.")
        
        # Inicializar motor y hacer consulta
        engine = RAGEngine(documents_dir="./test_data")
        response = engine.query("¿Qué es PyTorch?")
        print("\nRespuesta a '¿Qué es PyTorch?':")
        print(response)
        
        # Verificación
        assert "PyTorch" in response
        assert "aprendizaje profundo" in response