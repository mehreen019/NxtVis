from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI 
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import warnings
import os
#os.environ["OPENAI_API_KEY"] = ''
warnings.filterwarnings("ignore")
from llama_index.core import Settings

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

        # Initialize LLM
        self._llm = OpenAI(model="gpt-3.5-turbo", api_key=os.environ.get("OPENAI_API_KEY"))

        # Initialize Embedding Model
        self._embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Configure global settings
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model


        self._index = None
        self._create_kb()
        self._create_chat_engine()

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"D:\\coding\\projects\\NxtVis\\interactive_voice_bot\\rag\\airport_directions.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="airport_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def interact_with_llm(self, customer_query):
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
        You are an AI Airport Assistant at GlobalHub International Airport. Your tasks include:
        1. Help passengers find gates, services, and facilities
        2. Provide flight status updates
        3. Explain security procedures
        4. Guide to lounges, restaurants, and shops
        
        Always follow these rules:
        - First ask for flight number when relevant
        - Give directions with terminal numbers and landmarks
        - Keep responses under 15 words unless detailed instructions needed
        - Never invent information - say "I don't have that data" when unsure
        - Prioritize security regulations accuracy
        
        Ask these when appropriate (ONE question at a time):
        [Flight number? | Departure/Arrival? | Special needs? | Lounge access status?]
        """