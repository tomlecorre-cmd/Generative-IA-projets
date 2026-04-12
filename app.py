import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# On importe ton agent indestructible !
from RAG.agent import setup_agent

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Agent Financier IA", 
    page_icon="📈", 
    layout="centered"
)

st.title("📈 Assistant Financier IA")
st.markdown("Posez-moi vos questions sur la bourse, l'économie ou la stratégie de TotalEnergies. Je chercherai les infos en temps réel !")

# --- INITIALISATION DE L'AGENT ET DE L'HISTORIQUE ---
# On utilise st.session_state pour ne pas recharger l'agent à chaque question
if "agent" not in st.session_state:
    with st.spinner("🔌 Préchauffage du moteur LangGraph..."):
        st.session_state.agent = setup_agent()
        st.session_state.messages = [] # Pour stocker la conversation
        st.success("Agent prêt et connecté au Web !")

# Affichage de l'historique des messages précédents
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- BARRE DE CHAT ---
if prompt := st.chat_input("Ex: Quel est le cours de TTE.PA et multiplie-le par 500 ?"):
    
    # 1. On affiche la question de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # On la sauvegarde dans l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. On prépare les instructions pour l'IA
    system_prompt = (
         "Tu es un analyste financier de très haut niveau. Tu possèdes des outils spécialisés. "
         "RÈGLE 1 : Analyse la question pour choisir l'outil le plus précis. "
         "RÈGLE 2 : Tu peux enchaîner les outils. "
         "RÈGLE 3 : Réponds TOUJOURS en français."
    )
    
    inputs = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
    }

    # 3. On fait réfléchir l'Agent et on affiche sa réponse
    with st.chat_message("assistant"):
        with st.spinner("🤖 L'Agent réfléchit et utilise ses outils..."):
            try:
                final_response = ""
                
                # On fait tourner LangGraph
                for chunk in st.session_state.agent.stream(inputs, stream_mode="values"):
                    last_msg = chunk["messages"][-1]
                    
                    # On ne garde que la réponse finale de l'IA (on masque le code des outils)
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        final_response = last_msg.content
                
                # On affiche la réponse finale !
                st.markdown(final_response)
                
                # On sauvegarde la réponse dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                st.error(f"Une erreur est survenue dans la matrice : {e}")