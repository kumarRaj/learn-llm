# resolution_suggester.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import List, Dict
import requests
import json


class ResolutionSuggester:
    """
    End-to-end pipeline that:
    1. Embeds a new complaint
    2. Finds top 3 similar past tickets
    3. Sends them to an LLM
    4. Generates a suggested resolution
    """
    
    def __init__(self, embeddings_dir=None, llm_base_url="http://localhost:1234/v1"):
        """
        Initialize the ResolutionSuggester.
        
        Args:
            embeddings_dir: Directory containing pre-computed embeddings
            llm_base_url: Base URL for the LLM API endpoint
        """
        if embeddings_dir is None:
            embeddings_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.embeddings_dir = embeddings_dir
        self.project_root = os.path.dirname(embeddings_dir)
        self.llm_base_url = llm_base_url
        
        # Load embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test LLM connection
        self._test_llm_connection()
        
        # Load pre-computed embeddings and data
        self._load_embeddings()
        self._load_problems_and_solutions()
    
    def _test_llm_connection(self):
        """Test connection to the LLM server."""
        try:
            print(f"Testing connection to LLM at {self.llm_base_url}...")
            response = requests.get(f"{self.llm_base_url}/models", timeout=5)
            if response.status_code == 200:
                models = response.json().get('data', [])
                if models:
                    print(f"Connected! Available models: {[m.get('id') for m in models]}")
                    self.model_name = models[0]['id']
                else:
                    raise ConnectionError("No models available on LLM server")
            else:
                raise ConnectionError(f"LLM server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to LLM at {self.llm_base_url}. Make sure the server is running.")
        except Exception as e:
            raise ConnectionError(f"Error connecting to LLM: {e}")
    
    def _load_embeddings(self):
        """Load pre-computed embeddings from .npy files."""
        problem_emb_path = os.path.join(self.embeddings_dir, 'problem_embeddings.npy')
        ticket_ids_path = os.path.join(self.embeddings_dir, 'ticket_ids.npy')
        solution_emb_path = os.path.join(self.embeddings_dir, 'solution_embeddings.npy')
        
        try:
            self.problem_embeddings = np.load(problem_emb_path, allow_pickle=True)
            self.ticket_ids = np.load(ticket_ids_path, allow_pickle=True)
            self.solution_embeddings = np.load(solution_emb_path, allow_pickle=True)
            print(f"Loaded {len(self.problem_embeddings)} problem embeddings")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Embedding files not found in {self.embeddings_dir}. Error: {e}")
    
    def _load_problems_and_solutions(self):
        """Load original problem descriptions and solutions from CSV."""
        csv_path = os.path.join(self.project_root, 'customer_support_tickets.csv')
        
        try:
            df = pd.read_csv(csv_path)
            self.problems = df['Ticket Description'].tolist()
            self.solutions = df['Resolution'].tolist()
            
            # Handle NaN values
            self.problems = [str(p) if pd.notna(p) else "" for p in self.problems]
            self.solutions = [str(s) if pd.notna(s) else "" for s in self.solutions]
            
            print(f"Loaded {len(self.problems)} problems and solutions")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found at {csv_path}. Error: {e}")
    
    def _embed_text(self, text):
        """Create embedding for a single text."""
        inputs = self.embedding_tokenizer(str(text), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def find_similar_tickets(self, complaint: str, top_k: int = 3) -> List[Dict]:
        """
        Find top K similar tickets to the given complaint.
        Only returns tickets that have non-empty resolutions.
        
        Args:
            complaint: Customer complaint/problem statement
            top_k: Number of similar tickets to return
            
        Returns:
            List of dictionaries with ticket info and similarity scores
        """
        # Embed the complaint
        query_embedding = self._embed_text(complaint)
        
        # Reshape embeddings for comparison
        problem_embs_2d = np.vstack([emb.reshape(1, -1) for emb in self.problem_embeddings])
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding_2d, problem_embs_2d)[0]
        
        # Get indices sorted by similarity (highest first)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Filter for tickets with non-empty solutions and collect top K
        results = []
        for idx in sorted_indices:
            # Only include tickets with actual solutions
            if self.solutions[idx] and len(str(self.solutions[idx]).strip()) > 10:
                results.append({
                    'ticket_id': int(self.ticket_ids[idx]),
                    'similarity': float(similarities[idx]),
                    'problem': self.problems[idx],
                    'solution': self.solutions[idx]
                })
            
            # Stop once we have enough results
            if len(results) >= top_k:
                break
        
        if len(results) < top_k:
            print(f"Warning: Only found {len(results)} tickets with valid solutions (requested {top_k})")
        
        return results
    
    def _build_context_prompt(self, complaint: str, similar_tickets: List[Dict]) -> str:
        """
        Build a prompt with the complaint and similar past tickets.
        Forces the LLM to adapt from past solutions rather than generating new ones.
        
        Args:
            complaint: The new complaint
            similar_tickets: List of similar past tickets with their solutions
            
        Returns:
            Formatted prompt for the LLM
        """
        context = f"""You are a customer support assistant. Your job is to adapt and reuse proven solutions from past tickets.

NEW CUSTOMER COMPLAINT:
{complaint}

PAST SOLUTIONS THAT WORKED FOR SIMILAR ISSUES:
"""
        
        for i, ticket in enumerate(similar_tickets, 1):
            context += f"""
Solution #{i} (from Ticket #{ticket['ticket_id']}, Similarity: {ticket['similarity']:.2%}):
Past Problem: {ticket['problem'][:250]}
What We Did: {ticket['solution'][:250]}
---"""
        
        context += f"""

INSTRUCTIONS:
1. Analyze the new complaint and the past solutions above
2. Choose the MOST RELEVANT past solution
3. Adapt it specifically for this new complaint (change product names, details as needed)
4. Keep the core solution approach from the past ticket
5. Do NOT invent new solutions - only adapt what worked before
6. Format your response as: "We solved a similar issue by [ADAPTED SOLUTION]. Let's try this approach..."

ADAPTED RESOLUTION FOR THE NEW COMPLAINT:
"""
        
        return context
    
    def suggest_resolution(self, complaint: str, top_k: int = 3, max_length: int = 500) -> Dict:
        """
        Generate a suggested resolution for a new complaint.
        
        Args:
            complaint: The new customer complaint
            top_k: Number of similar tickets to use as context
            max_length: Maximum length of the generated resolution
            
        Returns:
            Dictionary containing:
                - complaint: The original complaint
                - similar_tickets: Top similar tickets used for context
                - suggested_resolution: LLM-generated resolution
        """
        print(f"Finding similar tickets for complaint...")
        similar_tickets = self.find_similar_tickets(complaint, top_k)
        
        print(f"Building context prompt...")
        prompt = self._build_context_prompt(complaint, similar_tickets)
        
        print(f"Generating resolution with LLM...")
        try:
            response = requests.post(
                f"{self.llm_base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful customer support assistant."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": max_length,
                    "top_p": 0.9
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"LLM API returned status code {response.status_code}: {response.text}")
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to LLM at {self.llm_base_url}")
        except Exception as e:
            raise Exception(f"Error calling LLM: {e}")
        
        return {
            'complaint': complaint,
            'similar_tickets': similar_tickets,
            'suggested_resolution': generated_text.strip(),
            'prompt_used': prompt
        }


if __name__ == '__main__':
    print("Initializing Resolution Suggester...")
    suggester = ResolutionSuggester()
    
    print("\n" + "="*100)
    print("Resolution Suggester - AI-Powered Customer Support")
    print("="*100)
    print("Type 'exit' or 'quit' to exit\n")
    
    while True:
        try:
            complaint = input("Enter customer complaint: ").strip()
            
            if complaint.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not complaint:
                print("Please enter a valid complaint.\n")
                continue
            
            print("\n" + "-"*100)
            result = suggester.suggest_resolution(complaint, top_k=3)
            
            print(f"\nCUSTOMER COMPLAINT:")
            print(f"{result['complaint']}\n")
            
            print(f"SIMILAR PAST TICKETS:")
            for i, ticket in enumerate(result['similar_tickets'], 1):
                print(f"\n{i}. Ticket #{ticket['ticket_id']} (Similarity: {ticket['similarity']:.2%})")
                print(f"   Problem: {ticket['problem'][:150]}")
                print(f"   Past Resolution: {ticket['solution'][:150]}")
            
            print(f"\nSUGGESTED RESOLUTION:")
            print(result['suggested_resolution'])
            
            print("\n" + "="*100 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
