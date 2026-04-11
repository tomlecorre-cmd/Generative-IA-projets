import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# On importe les documents 
from RAG.loader import load_and_split_documents

# Le dossier où sera sauvegardée la base
CHROMA_DB_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def create_vectorstore():
   
    print(" Démarrage du processus de vectorisation")
    
    chunks = load_and_split_documents()
    if not chunks:
        print("Impossible de créer la base : aucun chunk fourni.")
        return None

    print(f"Création des embeddings pour {len(chunks)} chunks")
    print("Téléchargement du modèle HuggingFace (la première fois seulement) et calculs")
    
    # Initialisation du modèle en local
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Création de la base de données
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )
    
    print(f" Base vectorielle ChromaDB créée et sauvegardée dans : {CHROMA_DB_DIRECTORY}")
    return vector_store

def get_vectorstore():
    
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIRECTORY, 
        embedding_function=embeddings
    )
    return vector_store

# --- ZONE DE TEST ---
if __name__ == "__main__":
    create_vectorstore()