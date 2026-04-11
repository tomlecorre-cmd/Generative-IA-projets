import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "data")

def load_and_split_documents():
    
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        print("Pas de dossier.")
        return None

    loader = PyPDFDirectoryLoader(DATA_DIRECTORY)
    documents = loader.load()

    if not documents:
        print("Aucun document PDF trouvé dans le dossier 'data/'.")
        return None

    print(f" Succès : {len(documents)} pages chargées depuis les PDF.")

    print("Découpage des documents en chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Terminé : {len(chunks)} chunks créés et prêts pour la vectorisation.")
    
    return chunks

# --- ZONE DE TEST ---
if __name__ == "__main__":
    
    my_chunks = load_and_split_documents()
    if my_chunks:
        print("\n--- Aperçu du premier chunk ---")
        print(my_chunks[0].page_content[:200] + "...")