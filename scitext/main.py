from scitext.models import load_models
from scitext.paper import Paper
from scitext.eval import evaluate_kg
from pathlib import Path

def main():
    refined, rebel = load_models()

    paper = Paper(Path('papers/35-hoehler.pdf'))
    kg = paper.extract_kg(refined, rebel)
    kg.save_rdf(paper.ttl_path)
    evaluate_kg(Path(paper.ttl_path), print_summary=True)

if __name__ == '__main__':
    main()