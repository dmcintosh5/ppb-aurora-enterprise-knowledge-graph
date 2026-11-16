import os
import sys
import subprocess

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.graph_orchestrator import process_corpus

def run_script(script_path):
    print(f"\n[RUNNING] {script_path}...")
    try:
        subprocess.run([sys.executable, script_path], check=True, text=True)
        print(f"[SUCCESS] {script_path} completed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {script_path} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    print("="*50)
    print("      PPB DAIS - AGENTIC DATA PIPELINE REBUILD      ")
    print("="*50)

    try:
        # 1. Clear Neo4j
        run_script('scripts/clear_neo4j.py')

        # 2. Generate Ontology (Ensure blueprint is current)
        run_script('scripts/ontology_generator.py')

        # 3. Execute Agentic Orchestrator
        # This single step now handles Ingestion, Embedding, and Ontology-Aligned Graph Building
        print("\n[*] Starting Agentic Graph Construction...")
        process_corpus()
        print("[SUCCESS] Agentic Orchestrator completed.")

        print("\n" + "="*50)
        print("PIPELINE COMPLETE: Knowledge Graph & Vector Store Rebuilt!")
        print("="*50)
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user. Partial data may be present in Neo4j.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Critical Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
