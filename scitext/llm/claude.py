import anthropic
import base64
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('../../.env')


def extract_from_web(pdf_url: str):
    """Fetch a PDF from the web and extract Triples using Claude.
    
    url: URL to PDF file, e.g. https://github.com/AlexanderProchnow/scitext/blob/main/Project_Report.pdf
    """
    import httpx
    pdf_data = base64.standard_b64encode(httpx.get(pdf_url).content).decode("utf-8")

    return extract(pdf_data)


def extract_from_file(pdf_path: Path):
    """Extract Triples from local PDF.

    pdf_path: Path to PDF on the local file system
    """
    with open(pdf_path, "rb") as pdf_file:
        binary_data = pdf_file.read()
        base_64_encoded_data = base64.b64encode(binary_data)
        pdf_data = base_64_encoded_data.decode('utf-8')

    return extract(pdf_data)


def extract(pdf_data):
    client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    message = client.beta.messages.create(
        model="claude-3-5-sonnet-20241022",
        betas=["pdfs-2024-09-25"],
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data
                        }
                    },
                    {
                        "type": "text",
                        "text": """
                        Extract the main entities and relations for a scientific Knowledge Graph.
                        Return only entity-relation-entity triples.
                        """
                    }
                ]
            }
        ],
    )

    return message.content[0].text
