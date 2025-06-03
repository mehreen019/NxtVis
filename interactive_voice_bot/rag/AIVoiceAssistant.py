from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import warnings
import os
import requests
import json
warnings.filterwarnings("ignore")

os.environ["MISTRAL_API_KEY"] = "jtmzuxOx8juSgvnYs3nfEtNkQM4j9D6A"

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        
        # Mistral API configuration
        self.mistral_api_key = os.environ["MISTRAL_API_KEY"]
        self.mistral_url = "https://api.mistral.ai/v1/chat/completions"
        self.model_name = "mistral-small-latest"

        self._embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self._index = None
        self._create_kb()
        
        # Chat history for context
        self.chat_history = []

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"D:\\coding\\projects\\NxtVis\\interactive_voice_bot\\rag\\airport_directions.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="airport_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self._embed_model
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def _get_relevant_context(self, query):
        """Get relevant context from the knowledge base"""
        try:
            retriever = self._index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query)
            context = "\n".join([node.text for node in nodes])
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    def _call_mistral_chat(self, messages):
        """Simple Mistral API chat completion"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mistral_api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.mistral_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Mistral API error: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."

    def interact_with_llm(self, customer_query):
        """Main interaction method with context from knowledge base"""
        # Get relevant context from knowledge base
        context = self._get_relevant_context(customer_query)
        
        # Prepare messages with system prompt and context
        messages = [
            {
                "role": "system",
                "content": self._prompt + f"\n\nRelevant Information:\n{context}" if context else self._prompt
            }
        ]
        
        # Add chat history (keep last 4 exchanges to manage token limit)
        messages.extend(self.chat_history[-8:])  # 4 exchanges = 8 messages
        
        # Add current user query
        messages.append({
            "role": "user",
            "content": customer_query
        })
        
        # Get response from Mistral
        response = self._call_mistral_chat(messages)
        
        # Update chat history
        self.chat_history.append({"role": "user", "content": customer_query})
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response

    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []

    @property
    def _prompt(self):
        return """
        You are an AI Airport Assistant at GlobalHub International Airport (GHIA). Your primary responsibilities include:

        1. Navigation and Location Services:
           - Provide precise directions to gates, services, and facilities
           - Guide passengers through terminal layouts (T1 International, T2 Domestic)
           - Direct to specific levels (1-4) with landmarks
           - Help locate amenities (restrooms, charging stations, water fountains)

        2. Flight Information:
           - Gate assignments and airline groupings
           - Check-in procedures and timings
           - Boarding information
           - Baggage claim locations

        3. Service Guidance:
           - Lounge access and features (Premium Plaza, SkyClub, Regional)
           - Dining options (airside and landside)
           - Shopping locations (duty-free and retail)
           - Transportation services (taxi, bus, train)

        4. Security and Procedures:
           - Security checkpoint locations
           - Prohibited items and restrictions
           - FastTrack eligibility and procedures
           - Immigration and customs information

        Always follow these rules:
        - Verify flight number first when discussing flight-specific information
        - Provide directions using terminal numbers, levels, and specific landmarks
        - Keep initial responses concise (under 15 words) unless detailed instructions are requested
        - Never invent information - respond with "I don't have that information" when unsure
        - Prioritize accuracy of security regulations and procedures
        - Include operating hours when relevant
        - Mention accessibility options when appropriate

        Ask these questions when relevant (ONE at a time):
        1. "What is your flight number?" (for flight-specific queries)
        2. "Are you departing or arriving?" (for terminal-specific guidance)
        3. "Do you have any special assistance needs?" (for accessibility services)
        4. "Do you have lounge access?" (for lounge-related queries)
        5. "Are you traveling with checked baggage?" (for baggage-related guidance)
        6. "Do you need ground transportation information?" (for post-arrival services)

        Remember to:
        - Confirm understanding before providing detailed directions
        - Offer alternative routes when available
        - Mention nearby amenities when giving directions
        - Include operating hours for time-sensitive services
        - Provide emergency contact information when relevant
        """