from scitext.models import load_models
from scitext.paper import Paper

def main():
    refined, rebel = load_models()

    paper = Paper()
    kg = paper.extract_kg(refined, rebel)
    kg.save_rdf(f"{paper.name}.ttl")

if __name__ == '__main__':
    main()