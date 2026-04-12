import os
import sys
import requests 
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage

from RAG.vectorstore import get_vectorstore

load_dotenv()
os.environ["USER_AGENT"] = "MonAssistantFinancier/1.0"

def setup_agent():
    print(" Déploiement via LangGraph")
    
    vector_store = get_vectorstore()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    @tool
    def recherche_documents_internes(query: str) -> str:
        """Outil PRIORITAIRE. Utilise-le pour chercher la stratégie de TotalEnergies, ou les rapports du FMI et de la Banque de France."""
        return retriever.invoke(query)

    @tool
    def api_calcul(expression: str) -> str:
        """Utilise cet outil pour faire des calculs mathématiques exacts via API. 
        L'entrée DOIT être une expression mathématique (ex: 500 * 100.5)."""
        try:
            # On envoie l'expression mathématique directement à l'API 
            response = requests.get(f"http://api.mathjs.org/v4/?expr={expression}")
            if response.status_code == 200:
                return response.text
            return "Erreur de l'API de calcul."
        except Exception as e:
            return f"Erreur de connexion : {e}"


    web_tool = DuckDuckGoSearchRun(name="recherche_internet_generale")
    finance_tool = YahooFinanceNewsTool(name="recherche_bourse_yahoo")
    arxiv_tool = ArxivQueryRun(name="recherche_theses_arxiv")

    tools = [recherche_documents_internes, web_tool, finance_tool, arxiv_tool, api_calcul]

    print("Initialisation de Llama 3.1...")
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor

if __name__ == "__main__":
    agent = setup_agent()
    
    print("\n" + "="*50)
    question = "Cherche sur Yahoo Finance le cours actuel de l'action de TotalEnergies (TTE.PA), puis utilise l'API de calcul pour multiplier ce prix par 500 actions."
    print(f"QUESTION : {question}")
    print("="*50 + "\n")
    
    system_prompt = (
         "Tu es un analyste financier de très haut niveau. Tu possèdes 5 outils spécialisés. "
         "RÈGLE 1 : Analyse la question pour choisir l'outil le plus précis. "
         "RÈGLE 2 : Tu peux enchaîner les outils. "
         "RÈGLE 3 : Réponds TOUJOURS en français."
    )

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
    }
    
    # Exécution !
    for chunk in agent.stream(inputs, stream_mode="values"):
        chunk["messages"][-1].pretty_print()