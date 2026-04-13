import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from RAG.vectorstore import get_vectorstore

load_dotenv()

def setup_rag_chain():
    
    print(" Connexion à la base vectorielle ChromaDB...")
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    print(" Initialisation de Llama 3 (via Groq)...")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    template = """Tu es un assistant expert en analyse économique et financière.
Utilise EXCLUSIVEMENT les éléments de contexte suivants pour répondre à la question.
Si la réponse ne se trouve pas dans le contexte, dis simplement que tu ne sais pas.

RÈGLE ABSOLUE : Tu dois OBLIGATOIREMENT citer le nom du document dont tu tires l'information.
RÈGLE ABSOLUE : Termine TOUJOURS ta réponse en gras par : **Source : [nom exact du fichier PDF]**

Contexte :
{context}

Question : {question}

Réponse :"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        result = ""
        for doc in docs:
            nom_fichier = doc.metadata.get('source', 'Inconnu').split("\\")[-1].split("/")[-1]
            result += f"[Source : {nom_fichier}]\n{doc.page_content}\n\n"
        return result

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

if __name__ == "__main__":
    print("\n--- Test du Pipeline RAG Multi-Documents ---")
    
    chain, retriever = setup_rag_chain()
    
    questions_test = [
        "Selon le rapport du FMI, quelles sont les prévisions pour la croissance de l'économie mondiale ?",
        "Que dit le document sur la stratégie de durabilité ou d'éthique de TotalEnergies ?",
        "Selon la Banque de France, quelles sont les projections pour l'inflation ou le chômage ?"
    ]
    
    for index, question in enumerate(questions_test):
        print("\n" + "="*50)
        print(f"QUESTION {index + 1} : {question}")
        print("="*50)
        
        docs_sources = retriever.invoke(question)
        reponse = chain.invoke(question)
        
        print("\n RÉPONSE :")
        print(reponse)
        
        print("\n SOURCES UTILISÉES PAR CHROMADB :")
        for i, doc in enumerate(docs_sources):
            source = doc.metadata.get('source', 'Inconnu')
            page = doc.metadata.get('page', 'Inconnu')
            nom_fichier = source.split("\\")[-1].split("/")[-1]
            print(f"  - Source {i+1} : {nom_fichier} (Page {page})")
            
    print("\n" + "="*50)
    print(" FIN DU TEST ")