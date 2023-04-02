from scitext.models import load_models
from scitext.paper import Paper
from scitext.eval import evaluate_kg
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="""Extract KG from scientific paper.
        Example: python -m scitext.main --paper papers/36-ramirez-llodra.pdf --evaluate""")
    parser.add_argument('--paper', type=str, default='papers/36-ramirez-llodra.pdf',
                         help="Path to PDF. Example: papers/36-ramirez-llodra.pdf")
    parser.add_argument('--folder', type=str, default=None,
                         help="Alternative to --paper: Path to folder containing papers in PDF format. Example: papers")
    parser.add_argument('--evaluate', action='store_true', default=False,
                         help="Evaluate KG using GPT-4. Requires an OpenAI API key. (Default: False)")
    args = parser.parse_args()

    refined, rebel = load_models()

    if args.folder:
        # all papers in folder
        paper_paths = Path(args.folder).glob('*.pdf')
    else:
        # single paper
        paper_paths = [Path(args.paper)]
    
    for paper_path in paper_paths:
        paper = Paper(paper_path)
        kg = paper.extract_kg(refined, rebel)
        kg.save_rdf(paper.ttl_path)
        if args.evaluate:
            evaluate_kg(Path(paper.ttl_path), print_summary=True)

if __name__ == '__main__':
    main()