# reading scientific data
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize
from scitext.models import Refined, Rebel
import kglab
import rdflib

class Paper(PdfReader):
    """A class to read a scientific paper and return a KG representing it.
    
    Inherits from PdfReader and thus can be used as such."""
    def __init__(self, path: Path=Path('papers/36-ramirez-llodra.pdf')):
        super().__init__(path)
        self.name = path.stem
        self.number_of_pages = len(self.pages)

    def get_text(self, page_num: int=1):
        """Extract text from a specific page."""
        return self.pages[page_num].extract_text()
    
    # TODO function to extract all images using page.images
    
    def extract_kg(self, refined: Refined, rebel: Rebel) -> kglab.KnowledgeGraph:
        """Extract knowledge graph from a paper."""
        
        kg = kglab.KnowledgeGraph()

        for page_num in tqdm(range(1, self.number_of_pages)):
            # Extract text from page
            text = self.get_text(page_num)
            

            # Extract relations from page using REBEL
            sentences = sent_tokenize(text)
            for i, sent in enumerate(sentences): #tqdm(enumerate(sentences), total=len(sentences)):
                # Preprocess sentence
                sent = "".join(ch for ch in sent if ch not in ['.', ',', '!', '?', '-', '(', ')', ':', ';', '\'', '\n'])

                # Extract entities from sentence using ReFinEd
                entities = pd.DataFrame(refined.process_text(sent))

                if entities.empty:
                    continue

                # Extract/predict relations from sentence using REBEL
                triples = rebel.predict(sent)

                for _, triple in triples.iterrows():
                    # Search for head and tail in extracted entities
                    head = triple['head']
                    tail = triple['tail']

                    head_match = entities[entities['text'].str.contains(head, case=False)]
                    tail_match = entities[entities['text'].str.contains(tail, case=False)]

                    if len(head_match) == 0 or len(tail_match) == 0:
                        continue

                    else:
                        # create entries in the final KG TODO add date handling
                        for _, h in head_match[head_match['coarse_type'] != 'DATE'].iterrows():
                            head_entry = rdflib.URIRef(h['predicted_entity'])
                            for _, t in tail_match[tail_match['coarse_type'] != 'DATE'].iterrows():
                                if t['coarse_type'] in ['QUANTITY', 'CARDINAL']:
                                    tail_entry = rdflib.Literal(t['text'])
                                else:
                                    tail_entry = rdflib.URIRef(t['predicted_entity'])

                                relation_entry = rebel.vocab[
                                    rebel.vocab['predicate'].str.match(triple['relation'])
                                ]
                                if relation_entry.empty:
                                    # save unfound relations to file
                                    with open('data/rebel_dataset/unfound_relations.txt', 'a') as f:
                                        f.write(triple['relation'] + '\n')
                                    continue

                                relation_entry = f'https://www.wikidata.org/wiki/Property:{relation_entry["uri"].iloc[0]}'
                                relation_entry = rdflib.URIRef(relation_entry)
                                kg.add(head_entry, relation_entry, tail_entry)

        return kg






            


