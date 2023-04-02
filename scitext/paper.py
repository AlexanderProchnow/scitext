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
        self.ttl_path = f"{self.name}.ttl"

    def get_text(self, page_num: int=1):
        """Extract text from a specific page."""
        return self.pages[page_num].extract_text()
    
    # TODO function to extract all images using page.images and add them to the KG
    
    def extract_kg(self, refined: Refined, rebel: Rebel, confidence_threshold: float=0.5) -> kglab.KnowledgeGraph:
        """Extract knowledge graph from a paper.
        
        confidence_threshold: minimum confidence score for an entity extracted by ReFinEd
        to be included in the KG."""
        
        kg = kglab.KnowledgeGraph()

        for page_num in tqdm(range(1, self.number_of_pages)):
            # Extract text from page
            text = self.get_text(page_num)

            # Extract relations from page using REBEL
            sentences = sent_tokenize(text)
            for i, sent in enumerate(sentences): #tqdm(enumerate(sentences), total=len(sentences)):
                # Preprocess sentence
                sent = "".join(ch for ch in sent if ch not in [
                    ',', '!', '?', '-', '(', ')', ':', ';', '\'', '\n'#, '.'
                ])

                # Extract entities from sentence using ReFinEd
                entities = pd.DataFrame(refined.process_text(sent))

                if entities.empty:
                    continue

                # filter entities by confidence, replace NaN with 1 (occurs e.g. by DATE)
                conf_scores = entities['entity_linking_model_confidence_score'].fillna(1)
                entities = entities[conf_scores > confidence_threshold]

                # Extract/predict relations from sentence using REBEL
                triples = rebel.predict(sent)

                self._match_and_add(kg, entities, triples, rebel)

        return kg


    def _match_and_add(self, kg: kglab.KnowledgeGraph, entities: pd.DataFrame, triples: pd.DataFrame,
                        rebel: Rebel):
        for _, triple in triples.iterrows():
            # Search for head and tail in extracted entities
            head = triple['head']
            tail = triple['tail']

            if head == tail:
                continue

            head_match = entities[entities['text'].str.contains(head, case=False)]
            tail_match = entities[entities['text'].str.contains(tail, case=False)]

            if len(head_match) == 0 or len(tail_match) == 0:
                continue

            # create entries in the final KG TODO add date handling
            for _, h in head_match[head_match['coarse_type'] != 'DATE'].iterrows():
                head_entry = self._to_URI(h, kg, head=True)
                if head_entry is None:
                    continue

                for _, t in tail_match[tail_match['coarse_type'] != 'DATE'].iterrows():
                    if t['coarse_type'] in ['QUANTITY', 'CARDINAL']:
                        tail_entry = rdflib.Literal(t['text'])
                    else:
                        tail_entry = self._to_URI(t, kg, head=False) #rdflib.URIRef(t['predicted_entity'])

                    relation_entry = rebel.vocab[
                        rebel.vocab['predicate'].str.match(triple['relation'])
                    ]
                    if relation_entry.empty:
                        # save unfound relations to file
                        with open('data/rebel_dataset/unfound_relations.txt', 'a') as f:
                            f.write(triple['relation'] + '\n')
                        continue

                    relation_entry = rdflib.URIRef(
                        # f'https://www.wikidata.org/wiki/Property:{relation_entry["uri"].iloc[0]}'
                        f'http://www.wikidata.org/prop/direct/{relation_entry["uri"].iloc[0]}' 
                    )
                    if head_entry == tail_entry:
                        continue
                    
                    print(head_entry, relation_entry, tail_entry)
                    kg.add(head_entry, relation_entry, tail_entry)


    def _to_URI(self, h: pd.Series, kg: kglab.KnowledgeGraph, head: bool) -> rdflib.URIRef:
        # if no entity predicted, take best candidate
        # TODO add confidence score of link prediction to KG
        if h['predicted_entity'] == None or h['predicted_entity']['wikidata_entity_id'] == None:
            if h['candidate_entities']:
                qid, confidence = h['candidate_entities'][0]
            else: 
                if not head:
                    node = rdflib.Literal(h['text'])
                    # TODO do always?
                    # add predicted class to entity, e.g. node a scientist.
                    # for this, node must be made a IRI
                    # if h['predicted_entity_types']:
                    #     qid, name, conf = h['predicted_entity_types'][0]
                    #     kg.add(node, rdflib.RDF.type, rdflib.URIRef(f'http://www.wikidata.org/wiki/{qid}'))
                    return node
                else: return None
        else:
            qid = h['predicted_entity']['wikidata_entity_id']

        # return rdflib.URIRef(f'https://www.wikidata.org/wiki/{qid}')
        return rdflib.URIRef(f'http://www.wikidata.org/entity/{qid}')
            


