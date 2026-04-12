import chainlit as cl
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from RAG.agent import setup_agent

@cl.on_chat_start
async def start_chat():
    agent = setup_agent()
    cl.user_session.set("agent", agent)
    
    system_prompt = (
         "Tu es un analyste financier de très haut niveau. Tu possèdes des outils spécialisés. "
         "RÈGLE 1 : Analyse la question pour choisir l'outil le plus précis. "
         "RÈGLE 2 : Tu peux enchaîner les outils. "
         "RÈGLE 3 : Réponds TOUJOURS en français."
    )
    cl.user_session.set("history", [SystemMessage(content=system_prompt)])
    
    await cl.Message(
        content="📈 **Agent Financier Connecté.** (Mémoire optimisée activée !)"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("history")
    
    history.append(HumanMessage(content=message.content))
    
    if len(history) > 5:
        history = [history[0]] + history[-4:]
    
    
    inputs = {"messages": history}

    msg = cl.Message(content="*L'Agent analyse et cherche...*")
    await msg.send()
    
    final_response = ""
    
    try:
        
        async for chunk in agent.astream(inputs, stream_mode="values"):
            last_msg = chunk["messages"][-1]
            
            if isinstance(last_msg, AIMessage) and last_msg.content:
                final_response = last_msg.content
                
        msg.content = final_response
        await msg.update()
        
       
        history.append(AIMessage(content=final_response))
        cl.user_session.set("history", history)
        
    except Exception as e:
        msg.content = f"❌ Oups, erreur de l'Agent : {e}"
        await msg.update()