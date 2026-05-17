# find_resolution.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import os


class ResolutionFinder:
    """Find similar problems and their resolutions using semantic embeddings."""
    
    def __init__(self, embeddings_dir=None):
        """
        Initialize the ResolutionFinder.
        
        Args:
            embeddings_dir: Directory containing pre-computed embeddings. 
                          Defaults to current script directory.
        """
        if embeddings_dir is None:
            embeddings_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.embeddings_dir = embeddings_dir
        self.project_root = os.path.dirname(embeddings_dir)
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load pre-computed embeddings and data
        self._load_embeddings()
        self._load_problems_and_solutions()
    
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
            raise FileNotFoundError(f"Embedding files not found in {self.embeddings_dir}. "
                                  f"Please run create_embeddings.py first. Error: {e}")
    
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
        """
        Create embedding for a single text.
        
        Args:
            text: String to embed
            
        Returns:
            Numpy array of shape (1, 384) containing the embedding
        """
        inputs = self.tokenizer(str(text), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def find_resolution(self, problem_statement, top_k=5):
        """
        Find top K similar problems and their resolutions.
        
        Args:
            problem_statement: String describing the problem
            top_k: Number of similar problems to return (default: 5)
            
        Returns:
            List of dictionaries containing:
                - ticket_id: Original ticket ID
                - similarity: Cosine similarity score (0-1)
                - problem: Original problem description
                - solution: Resolution for the problem
        """
        # Create embedding for the input problem
        query_embedding = self._embed_text(problem_statement)
        
        # Reshape problem embeddings for comparison
        # Each embedding is (1, 384), reshape to (-1, 384) for comparison
        problem_embs_2d = np.vstack([emb.reshape(1, -1) for emb in self.problem_embeddings])
        query_embedding_2d = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity with all problems
        similarities = cosine_similarity(query_embedding_2d, problem_embs_2d)[0]
        
        # Get top K indices
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            results.append({
                'ticket_id': int(self.ticket_ids[idx]),
                'similarity': float(similarities[idx]),
                'problem': self.problems[idx],
                'solution': self.solutions[idx]
            })
        
        return results
    
    def find_resolution_batch(self, problem_statements, top_k=5):
        """
        Find top K similar problems for multiple problem statements.
        
        Args:
            problem_statements: List of problem description strings
            top_k: Number of similar problems to return per query
            
        Returns:
            List of lists, where each inner list contains results from find_resolution()
        """
        results = []
        for problem in problem_statements:
            results.append(self.find_resolution(problem, top_k))
        return results


if __name__ == '__main__':
    # Example usage
    finder = ResolutionFinder()
    
    print("\n" + "="*100)
    print("Resolution Finder - Find Similar Support Tickets and Resolutions")
    print("="*100)
    print("Type 'exit' or 'quit' to exit the program")
    print("="*100 + "\n")
    
    while True:
        try:
            # Get user input
            problem_statement = input("Enter a problem statement: ").strip()
            
            # Check for exit commands
            if problem_statement.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break
            
            # Skip empty input
            if not problem_statement:
                print("Please enter a valid problem statement.\n")
                continue
            
            print(f"\nSearching for similar problems...\n")
            
            # Find similar problems
            results = finder.find_resolution(problem_statement, top_k=5)
            
            print("Top 5 Similar Problems and Resolutions:")
            print("-" * 100)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Ticket ID: {result['ticket_id']} (Similarity: {result['similarity']:.4f})")
                print(f"   Problem: {result['problem'][:150]}")
                if result['solution']:
                    print(f"   Solution: {result['solution'][:150]}")
                else:
                    print(f"   Solution: (No solution recorded)")
            
            print("\n" + "="*100 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")
