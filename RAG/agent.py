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
        docs = retriever.invoke(query)
        result = ""
        for doc in docs:
            nom_fichier = doc.metadata.get('source', 'Inconnu').split("\\")[-1].split("/")[-1]
            result += f"[Source : {nom_fichier}]\n{doc.page_content}\n\n"
        return result

    @tool
    def calculatrice(expression: str) -> str:
        """Effectue des calculs mathématiques financiers précis.
        L'entrée DOIT être une expression Python valide (ex: '500 * 48.30' ou 'round(1234.5678, 2)').
        Opérations disponibles : +, -, *, /, round(), abs(), pow()."""
        try:
            fonctions_autorisees = {
                'round': round,
                'abs': abs,
                'pow': pow,
                'min': min,
                'max': max,
            }
            result = eval(expression, {"__builtins__": {}}, fonctions_autorisees)
            return f"{result:,.4f}"
        except Exception as e:
            return f"Erreur de calcul : {e}"

    @tool
    def meteo(ville: str) -> str:
        """Obtenir la météo actuelle d'une ville. Entrée : nom de la ville en français ou anglais (ex: 'Paris', 'Londres')."""
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={ville}&count=1&language=fr"
            geo_response = requests.get(geo_url)
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                return f"Ville '{ville}' introuvable."
            
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            nom_ville = geo_data["results"][0]["name"]
            
            meteo_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,weathercode&timezone=auto"
            meteo_response = requests.get(meteo_url)
            meteo_data = meteo_response.json()
            
            current = meteo_data["current"]
            temperature = current["temperature_2m"]
            vent = current["wind_speed_10m"]
            
            return f"Météo à {nom_ville} : {temperature}°C, vent à {vent} km/h."
        
        except Exception as e:
            return f"Erreur météo : {e}"

    web_tool = DuckDuckGoSearchRun(name="recherche_internet_generale")
    finance_tool = YahooFinanceNewsTool(name="recherche_bourse_yahoo")
    arxiv_tool = ArxivQueryRun(name="recherche_theses_arxiv")

    tools = [recherche_documents_internes, web_tool, finance_tool, arxiv_tool, calculatrice, meteo]

    print("Initialisation de Llama 3.1 8b...")
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor

if __name__ == "__main__":
    agent = setup_agent()
    
    print("\n" + "="*50)
    question = "Quelle est la météo à Paris aujourd'hui ?"
    print(f"QUESTION : {question}")
    print("="*50 + "\n")
    
    system_prompt = (
        "Tu es un analyste financier de très haut niveau. Tu possèdes 6 outils spécialisés. "
        "RÈGLE 1 : Analyse la question pour choisir l'outil le plus précis. "
        "RÈGLE 2 : Tu peux enchaîner les outils si nécessaire. "
        "RÈGLE 3 : Réponds TOUJOURS en français. "
        "RÈGLE 4 : Cite TOUJOURS tes sources en gras : **Source : [nom du fichier]**"
    )

    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
    }
    
    for chunk in agent.stream(inputs, stream_mode="values"):
        chunk["messages"][-1].pretty_print()