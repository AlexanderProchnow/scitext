# reading scientific data
import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize

class Paper(PdfReader):
    """A class to read a scientific paper and return a KG representing it.
    
    Inherits from PdfReader and thus can be used as such."""
    def __init__(self, path: Path=Path('papers/36-ramirez-llodra.pdf')):
        super().__init__(path)
        self.number_of_pages = len(self.pages)

    def get_text(self, page_num: int=1):
        """Extract text from a specific page."""
        return self.pages[page_num].extract_text()
    
    # TODO function to extract all images using page.images
    
    def extract_kg(self, refined, rebel):
        """Extract knowledge graph from a paper."""
        
        for page_num in tqdm(range(1, self.number_of_pages), total=self.number_of_pages):
            text = self.get_text(page_num)
            # page_spans = refined.process_text(text)
            


    