"""
IISc Sharodiya Durgotsab — End-to-End Pipeline
Runs: Web Scraper → Knowledge Graph Builder → Graph RAG Question Answering Demo
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load .env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "scraped_data.json")
    graph_json = os.path.join(project_dir, "graph.json")
    graph_graphml = os.path.join(project_dir, "graph.graphml")

    print("\n" + "=" * 75)
    print("IISc Sharodiya Durgotsab — Graph RAG Knowledge Pipeline")
    print("Target: https://iiscsharodiyadurgotsab.vercel.app")
    print("=" * 75)

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Scrape Website
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("STEP 1: Scraping all 14 website pages & extracting structured entities...")
    print("=" * 75)
    from scraper import scrape_all

    data = scrape_all()
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nScraped data successfully saved to: {data_path}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Build Knowledge Graph
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("STEP 2: Building Knowledge Graph (Neo4j / NetworkX)...")
    print("=" * 75)
    from graph_builder import build_graph

    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    build_graph(data_path, neo4j_uri, neo4j_user, neo4j_password)

    # Export content.md
    print("\nExporting content.md...")
    try:
        import subprocess
        subprocess.run([sys.executable, "export_content_md.py"], check=True)
    except Exception as e:
        print(f"Note exporting content.md: {e}")

    # Build ChromaDB Vector Store
    print("\n" + "=" * 75)
    print("STEP 2B: Indexing into ChromaDB Vector Database...")
    print("=" * 75)
    try:
        from vector_db import build_vector_store
        build_vector_store()
    except Exception as e:
        print(f"Note on ChromaDB indexing: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Graph RAG Question Answering Demo
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("STEP 3: Demonstrating Graph RAG Retrieval & LLM Question Answering...")
    print("=" * 75)
    from graph_rag import GraphRAGRetriever, GraphRAGQA, demo_queries

    if neo4j_uri:
        retriever = GraphRAGRetriever(
            backend="neo4j",
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
    elif os.path.exists(graph_json):
        retriever = GraphRAGRetriever(backend="networkx", graph_path=graph_json)
    elif os.path.exists(graph_graphml):
        retriever = GraphRAGRetriever(backend="networkx", graph_path=graph_graphml)
    else:
        print("Graph file not found!")
        sys.exit(1)

    qa = GraphRAGQA(retriever)
    demo_queries(qa)
    retriever.close()

    print("\n" + "=" * 75)
    print("Pipeline Execution Completed Successfully!")
    print(f"  Scraped Data: {data_path}")
    print(f"  Graph JSON:   {graph_json}")
    print(f"  Graph GraphML:{graph_graphml}")
    print("  To start interactive Q&A mode, run: python graph_rag.py")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
