import pandas as pd
import rdflib
from pathlib import Path
import os
import openai
from tqdm import tqdm
import argparse

def evaluate_kg(path_to_kg: Path, print_summary: bool=False):
    # Load KG
    kg = rdflib.Graph()
    kg.parse(path_to_kg)#, format='turtle')

    out_file = Path(path_to_kg.stem + '_eval.csv')
    if out_file.exists():
        print('Loading existing evaluation file.')
        df = pd.read_csv(out_file)
    
    else:
        # query KG
        try: 
            qres = kg.query(
                """SELECT DISTINCT ?s ?slabel ?p ?plabel ?o ?olabel 
                WHERE {
                    ?s ?p ?o .
                    SERVICE <https://query.wikidata.org/sparql> {
                        ?s rdfs:label ?slabel.
                        FILTER(LANG(?slabel) = "en").
                        OPTIONAL {
                                ?o rdfs:label ?olabel.
                                FILTER(LANG(?olabel) = "en").
                        }
                    }
                }
                """
            )

            # save results to dataframe
            df = pd.DataFrame(qres, columns=['s', 'slabel', 'p', 'plabel', 'o', 'olabel'])

        except Exception as e:
            print(f'Please try again another time. Querying Wikidata to retrieve the rdfs:labels failed with: {e}')
            # raise e
            raise SystemExit(0)
        
        df.to_csv(out_file, index=False)

    # add plabel from vocab
    vocab = pd.read_csv('data/rebel_dataset/rebel_vocab.csv')
    df['plabel'] = df['p'].apply(lambda x: vocab[vocab['uri'] == x.split('/')[-1]]['predicate'].values[0])

    # send to GPT4
    openai.api_key = os.getenv("OPENAI_API_KEY")
    instructions = "You can evaluate a triple containing a <subject>, a <predicate> and an <object> based on if it makes sense in the real world and answer only with yes or no."
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if 'evaluation' in df.columns:
            if df.loc[i, 'evaluation'] == 'yes' or df.loc[i, 'evaluation'] == 'no':
                continue

        olabel = f"<{row['olabel']}>" if pd.notna(row['olabel']) else f"\"{row['o']}\""
        prompt = f"<{row['slabel']}> <{row['plabel']}> {olabel}"

        completion = openai.ChatCompletion.create(
            model="gpt-4", # or "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        pred = completion.choices[0].message.content
        pred = pred.lower()
        print(f"<{row['slabel']}> <{row['plabel']}> {olabel} => {pred}")
        # add prediction to dataframe
        df.loc[i, 'evaluation'] = pred

        # store at each step to avoid losing progress
        df.to_csv(out_file, index=False)

    if print_summary:
        val_counts = df['evaluation'].value_counts()
        print(val_counts, '\n')

        yes = val_counts['yes']
        no = val_counts['no']
        num_triples = yes+no
        perc_correct = yes/num_triples
        print(f'# triples: {num_triples}, % correct:  {perc_correct*100:.1f} %')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ttl', type=str, default='results/protein-delivery.ttl',
                            help="Path to KG in Turtle format. Example: results/protein-delivery.ttl")
    parser.add_argument('--folder', type=str, default=None,
                            help="Path to folder containing KGs in Turtle format. Example: results")
    args = parser.parse_args()

    if args.folder is None:
        evaluate_kg(Path(args.ttl), print_summary=True)
        
    else:
        for kg_path in Path(args.folder).glob('*.ttl'):
            print(kg_path.name.removesuffix('.ttl'))
            evaluate_kg(kg_path, print_summary=True)
            print()