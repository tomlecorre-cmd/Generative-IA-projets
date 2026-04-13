# 📈 Assistant Financier Intelligent — RAG + Agents

Projet réalisé dans le cadre du **DU Data Analytics — IA Générative 2026**

## 📌 Description

Un assistant conversationnel multi-compétences capable de :
- Répondre à des questions basées sur des documents financiers internes (RAG)
- Utiliser des outils externes de manière autonome (Agent)
- Maintenir une conversation contextuelle (Mémoire)

## 🏗️ Architecture

Question utilisateur
↓
Routeur (LLM)
↙     ↓      ↘
RAG   Agent   Chat
↘     ↓      ↙
Réponse finale

## 🛠️ Stack technique

- **LLM** : Llama 3 via Groq (gratuit et ultra-rapide)
- **Agent** : LangGraph ReAct
- **RAG** : LangChain + ChromaDB + HuggingFace Embeddings
- **Interface** : Chainlit

## 🔧 Outils disponibles

1. 📄 Recherche dans les documents internes (RAG)
2. 📊 Yahoo Finance (cours boursiers en temps réel)
3. 🔍 DuckDuckGo (recherche web)
4. 📚 ArXiv (publications scientifiques)
5. 🧮 Calculatrice financière (Python natif)

## ⚙️ Installation

1. Cloner le repo :
```bash
git clone https://github.com/tomlecorre-cmd/Generative-IA-projets
cd Generative-IA-projets
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Remplir .env avec votre clé GROQ_API_KEY
```

4. Créer la base vectorielle :
```bash
python RAG/vectorstore.py
```

## 🚀 Lancement
```bash
chainlit run app.py -w
```

## 📁 Structure du projet

├── app.py              # Interface Chainlit + Routeur
├── RAG/
│   ├── agent.py        # Agent LangGraph + outils
│   ├── qa_chain.py     # Pipeline RAG
│   ├── vectorstore.py  # Base ChromaDB
│   └── loader.py       # Chargement des PDF
├── data/               # Documents PDF
├── .env.example        # Variables d'environnement
└── requirements.txt    # Dépendances