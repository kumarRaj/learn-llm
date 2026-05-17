# create_embeddings.py
# Dataset: Customer Support Tickets
# Source: https://www.kaggle.com/datasets/sudipta0811/customer-support-ticket-dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

def create_embeddings(texts):
    """
    Create embeddings for a list of texts using sentence-transformers.
    
    Args:
        texts: List of strings to embed
    
    Returns:
        List of numpy arrays containing embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    
    embeddings = []
    for text in texts:
        # Handle None or NaN values
        if pd.isna(text):
            text = ""
        
        inputs = tokenizer(str(text), padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    
    return embeddings

if __name__ == '__main__':
    # Load customer support tickets
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'customer_support_tickets.csv')
    
    df = pd.read_csv(csv_path)
    
    # Extract problem and solution fields
    problems = df['Ticket Description'].tolist()
    solutions = df['Resolution'].tolist()
    
    print(f"Embedding {len(problems)} problems and solutions...")
    
    # Create embeddings for problems
    problem_embeddings = create_embeddings(problems)
    print(f"Created {len(problem_embeddings)} problem embeddings")
    
    # Create embeddings for solutions
    solution_embeddings = create_embeddings(solutions)
    print(f"Created {len(solution_embeddings)} solution embeddings")
    
    # Combine ticket IDs with embeddings for reference
    results = {
        'ticket_ids': df['Ticket ID'].tolist(),
        'problem_embeddings': problem_embeddings,
        'solution_embeddings': solution_embeddings
    }
    
    # Save embeddings to files
    np.save('problem_embeddings.npy', np.array(problem_embeddings, dtype=object), allow_pickle=True)
    np.save('solution_embeddings.npy', np.array(solution_embeddings, dtype=object), allow_pickle=True)
    
    # Save ticket IDs for reference
    ticket_ids = np.array(df['Ticket ID'].tolist())
    np.save('ticket_ids.npy', ticket_ids)
    
    print("Embeddings saved successfully!")
    print(f"  - problem_embeddings.npy")
    print(f"  - solution_embeddings.npy")
    print(f"  - ticket_ids.npy")