import pandas as pd
import rdflib
from pathlib import Path
import os
import openai
from tqdm import tqdm

def evaluate_kg(path_to_kg: Path, print_summary: bool=False):
    # Load KG
    kg = rdflib.Graph()
    kg.parse(path_to_kg)#, format='turtle')

    out_file = path_to_kg.stem + '_eval.csv'

    # query KG
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
    # df.to_csv('ramirez_eval_offline.csv', index=False)

    # add plabel from vocab
    vocab = pd.read_csv('data/rebel_dataset/rebel_vocab.csv')
    df['plabel'] = df['p'].apply(lambda x: vocab[vocab['uri'] == x.split('/')[-1]]['predicate'].values[0])

    # send to GPT4
    openai.api_key = os.getenv("OPENAI_API_KEY")
    instructions = "You can evaluate a triple containing a <subject>, a <predicate> and an <object> based on if it makes sense in the real world and answer only with yes or no."
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        olabel = f"<{row['olabel']}>" if pd.notna(row['olabel']) else f"\"{row['o']}\""
        prompt = f"<{row['slabel']}> <{row['plabel']}> {olabel}"

        completion = openai.ChatCompletion.create(
            model="gpt-4",#"gpt-3.5-turbo",
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

    df.to_csv(out_file, index=False)

    if print_summary:
        print(df['evaluation'].value_counts())


if __name__ == '__main__':
    evaluate_kg(Path('results/35-arrigo.ttl'), print_summary=True)