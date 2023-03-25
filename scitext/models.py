import pandas as pd
from refined.inference.processor import Refined

class Rebel():
    """A class to extract triples from text using REBEL from Babelscape."""
    def __init__(self, num_triples: int=5):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
        self.gen_kwargs = {
            "max_length": 256,
            "length_penalty": 0,
            "num_beams": num_triples,
            "num_return_sequences": num_triples
        }

        self.vocab = pd.read_csv('data/rebel_dataset/rebel_vocab.csv')


    def predict(self, sent: str) -> pd.DataFrame:
        """Predict triples from a sentence.
        
        Code adapted from the REBEL documentation."""

        model_inputs = self.tokenizer(sent, max_length=256, padding=True,
                                       truncation=True, return_tensors = 'pt')

        # Generate
        generated_tokens = self.model.generate(
            model_inputs["input_ids"].to(self.model.device),
            attention_mask=model_inputs["attention_mask"].to(self.model.device),
            **self.gen_kwargs,
        )

        # Extract text
        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)


        # Extract triplets
        rel_df = pd.DataFrame()
        for _, sentence in enumerate(decoded_preds):
            triplets = self._extract_triplets(sentence)
            for triple in triplets:

                triple_entry = pd.DataFrame({
                    'head': [triple['head']], 
                    'relation': [triple['type']], 
                    'tail': [triple['tail']], 
                    # 'page_number': [page_num], 
                    # 'sentence_number': [sent_num]
                })
                rel_df = pd.concat([rel_df, triple_entry])

        return rel_df
    

    def _extract_triplets(self, text):
        """Extract triplets from decoded predictions.
        
        Code taken from the REBEL documentation."""

        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        return triplets
    


def load_models() -> tuple[Rebel, Refined]:
    """Load models for extracting KGs from text.
    
    This may take up to 1h in total (around 10min for REBEL and 50min for ReFinEd)."""
    
    # load ReFinEd
    refined = Refined.from_pretrained(
        model_name='wikipedia_model_with_numbers',
        entity_set="wikipedia"
    )

    # load REBEL
    rebel = Rebel()

    return refined, rebel 