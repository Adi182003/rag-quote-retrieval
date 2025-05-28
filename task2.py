import streamlit as st
import pandas as pd
import faiss
import json
from sentence_transformers import SentenceTransformer

# Load artifacts
df = pd.read_pickle("quotes_df.pkl")
faiss_index = faiss.read_index("quotes.index")
model = SentenceTransformer('fine_tuned_model')

def semantic_search(query, top_k=5):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        results.append({
            "quote": df.iloc[idx]['quote'],
            "author": df.iloc[idx]['author'],
            "tags": df.iloc[idx]['tags'],
            "score": float(dist)
        })
    return results

def rag_generate(query, retrieved_quotes):
    context = "\n".join([f'"{r["quote"]}" - {r["author"]}' for r in retrieved_quotes])
    summary = f"Context for '{query}':\n{context}"
    return summary

def rag_evaluate(query, retrieved, summary):
    return {"answer_relevancy": "N/A", "faithfulness": "N/A", "context_precision": "N/A"}

def run_streamlit():
    st.title("RAG-Based Semantic Quote Retrieval")
    st.write("Enter a natural language query to retrieve relevant quotes, authors, and tags.")

    query = st.text_input("Your query", "quotes about hope by oscar wilde")
    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        with st.spinner("Retrieving..."):
            results = semantic_search(query, top_k=top_k)
            summary = rag_generate(query, results)
            evaluation = rag_evaluate(query, results, summary)

            st.subheader("Retrieved Quotes")
            for r in results:
                st.markdown(f"> {r['quote']}  \n**Author:** {r['author']}  \n**Tags:** {', '.join(r['tags'])}  \n**Score:** {r['score']:.4f}")

            st.subheader("Summary / Context")
            st.code(summary)

            st.subheader("RAG Evaluation (Placeholder)")
            st.json(evaluation)

            st.download_button(
                label="Download JSON Results",
                data=json.dumps(results, indent=2),
                file_name="quotes_results.json",
                mime="application/json"
            )

if __name__ == "__main__":
    run_streamlit()