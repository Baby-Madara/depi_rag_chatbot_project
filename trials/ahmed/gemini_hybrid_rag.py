from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from collections import Counter

# 1. Setup Models
llm = AzureChatOpenAI(azure_deployment="gpt-4", api_version="2024-02-01")
embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-small")

# 2. Prepare Hierarchical Data (Manual example of what Doc Intelligence output looks like)
# Each chunk knows its "lineage" in the metadata
docs = [
    Document(page_content="The cooling system uses liquid nitrogen.", 
            metadata={"path": "Engine > Cooling", "id": 1}),
    Document(page_content="Liquid nitrogen increases fuel volatility.", 
            metadata={"path": "Engine > Fuel System", "id": 2}),
    Document(page_content="Higher fuel volatility improves efficiency.", 
            metadata={"path": "Performance > Efficiency", "id": 3}),
]

vector_db = FAISS.from_documents(docs, embeddings)

# 3. The "Intersection" Logic
def intersected_retrieval(query, k=3):
    # Generate sub-questions to find intersecting evidence
    sub_questions = llm.invoke(f"Break this into 3 specific search queries: {query}").content.split("\n")
    
    all_results = []
    for q in sub_questions:
        # Get top matches for each sub-aspect of the question
        hits = vector_db.similarity_search(q, k=2)
        all_results.extend(hits)

    # Count how many times a specific hierarchy path or ID appears
    # Chunks that appear in multiple "sub-query" results are the INTERSECTIONS
    path_counts = Counter([doc.metadata['path'] for doc in all_results])
    
    # Sort results: prioritized by how many sub-queries they satisfied
    intersected_docs = sorted(all_results, key=lambda x: path_counts[x.metadata['path']], reverse=True)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_docs = [d for d in intersected_docs if not (d.metadata['id'] in seen or seen.add(d.metadata['id']))]
    
    return unique_docs[:k]

# 4. Execute
query = "How does nitrogen cooling impact overall efficiency?"
context = intersected_retrieval(query)

for doc in context:
    print(f"[{doc.metadata['path']}]: {doc.page_content}")
