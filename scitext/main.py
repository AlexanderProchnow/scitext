from scitext.models import load_models
from scitext.paper import Paper
from scitext.eval import evaluate_kg
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="""Extract KG from scientific paper.
        Example: python -m scitext.main --paper papers/36-ramirez-llodra.pdf""")
    parser.add_argument('--paper', type=str, default='papers/36-ramirez-llodra.pdf',
                         help="Path to PDF. Example: papers/36-ramirez-llodra.pdf")
    args = parser.parse_args()

    refined, rebel = load_models()

    paper = Paper(Path(args.paper))
    kg = paper.extract_kg(refined, rebel)
    kg.save_rdf(paper.ttl_path)
    evaluate_kg(Path(paper.ttl_path), print_summary=True)

if __name__ == '__main__':
    main()