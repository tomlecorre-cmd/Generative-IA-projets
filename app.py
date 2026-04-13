import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from RAG.agent import setup_agent
from RAG.qa_chain import setup_rag_chain

load_dotenv()

SYSTEM_PROMPT = (
    "Tu es un analyste financier de très haut niveau. Tu possèdes des outils spécialisés. "
    "RÈGLE 1 : Analyse la question pour choisir l'outil le plus précis. "
    "RÈGLE 2 : Tu peux enchaîner les outils si nécessaire. "
    "RÈGLE 3 : Réponds TOUJOURS en français. "
    "RÈGLE 4 : Cite TOUJOURS tes sources en gras : **Source : [nom du fichier]**"
)

def classify_question(question: str, llm) -> str:
    prompt = f"""Tu es un routeur. Classifie cette question en UN SEUL mot parmi : rag, agent, chat.

- rag : question sur des documents financiers internes (TotalEnergies, FMI, Banque de France, rapports, projections, stratégie)
- agent : question nécessitant un outil externe (calcul, bourse, recherche web, actualités, arxiv)
- chat : salutation ou question générale qui ne nécessite ni document ni outil

Question : "{question}"

Réponds uniquement par : rag, agent, ou chat"""

    response = llm.invoke(prompt)
    result = response.content.strip().lower()

    if "rag" in result:
        return "rag"
    elif "agent" in result:
        return "agent"
    else:
        return "chat"

@cl.on_chat_start
async def start_chat():
    agent = setup_agent()
    rag_chain, _ = setup_rag_chain()
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    cl.user_session.set("agent", agent)
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("llm", llm)
    cl.user_session.set("history", [SystemMessage(content=SYSTEM_PROMPT)])

    await cl.Message(
        content=(
            "📈 **Assistant Financier Intelligent**\n\n"
            "Bonjour ! Je peux vous aider à :\n"
            "- 📄 Répondre sur les rapports **FMI, Banque de France, TotalEnergies**\n"
            "- 📊 Consulter des **cours boursiers** en temps réel\n"
            "- 🔍 Effectuer des **recherches web** et scientifiques\n"
            "- 🧮 Réaliser des **calculs financiers**\n\n"
            "Posez votre question !"
        )
    ).send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    rag_chain = cl.user_session.get("rag_chain")
    llm = cl.user_session.get("llm")
    history = cl.user_session.get("history")

    user_question = message.content
    history.append(HumanMessage(content=user_question))

    # Mémoire glissante
    if len(history) > 5:
        history = [history[0]] + history[-4:]

    # Routage de la question
    route = classify_question(user_question, llm)

    msg = cl.Message(content="")
    await msg.send()

    try:
        if route == "rag":
            msg.content = "🔍 *Recherche dans les documents internes...*"
            await msg.update()
            response = rag_chain.invoke(user_question)
            msg.content = response

        elif route == "agent":
            msg.content = "⚙️ *L'agent sélectionne et utilise un outil...*"
            await msg.update()
            inputs = {"messages": history}
            final_response = ""
            async for chunk in agent.astream(inputs, stream_mode="values"):
                last = chunk["messages"][-1]
                if isinstance(last, AIMessage) and last.content:
                    final_response = last.content
            msg.content = final_response

        else:
            msg.content = "💬 *Réflexion en cours...*"
            await msg.update()
            response = llm.invoke(history)
            msg.content = response.content

        await msg.update()
        history.append(AIMessage(content=msg.content))
        cl.user_session.set("history", history)

    except Exception as e:
        msg.content = f"❌ Erreur : {e}"
        await msg.update()