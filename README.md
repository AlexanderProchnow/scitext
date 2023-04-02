# SciText

An automatic Text-to-Knowledge-Graph pipeline that was built to create knowledge graphs 
from scientific papers and evaluate them automatically.

This project was conducted as part of the Building and Mining Knowledge Graphs course
at Maastricht University. The project report can be found [here](https://github.com/AlexanderProchnow/scitext/blob/main/Project_Report.pdf).

## First use

On first use, the required models are downloaded, which may take up to 1h in total
(around 10min for REBEL and 50min for ReFinEd).

Before first use, install the package requirements:
```{bash}
python -m venv kg-env
source kg-env/bin/activate
pip install -r requirements.txt
```

## Use

To run this pipeline on a single paper:
```{python}
python -m scitext.main --paper <path_to_pdf> [--evaluate]
```
This returns a knowledge graph in Turtle format.

<br>

To run this pipeline on a folder containing multiple PDF files:
```{python}
python -m scitext.main --folder <path_to_folder> [--evaluate]
```

Using the optional --evaluate flag 
requires an OpenAI API key set as an environment variable, and returns
an evaluation CSV file containing an evaluation generated by GPT-4 (however GPT-3.5 can be used as well)
of each triple in the knowledge graph. Since the evaluation module both queries Wikidata and
sends requests GPT-4, there may be connection interruptions or the GPT-4 API may be overloaded
with other requests. In both cases, executing the evaluation module separately after waiting a few
seconds can solve this issue. To do this, run the following command.

Running evaluation separately on a single knowledge graph:
```{python}
python scitext/eval.py --ttl <path_to_turtle_kg>
```

Running evaluation separately on a folder containing multiple knowledge graphs:
```{python}
python scitext/eval.py --folder <path_to_folder>
```

For example usages, inspect the help description of a command using `-h`.

## Repo overview
`data/rebel_dataset/rebel_vocab.csv` contains the vocabulary used to map extracted relations to their rdfs:labels.

`papers` contains all papers used to evaluate the pipeline

`results` contains both the extracted knowledge graphs in Turtle format as well as the evaluation CSV files resulting from the automatic evaluation for each paper in the `papers` folder.

`scitext` contains the pipeline code, with each function's docstring explaining it's functionality.

`mapping_dict.ipynb` contains the code used to generate the `rebel_vocab.csv`, with the required REBEL dataset downloaded from within the `rebel_dataset.zip` at https://huggingface.co/datasets/Babelscape/rebel-dataset/tree/main .

`requirements.txt` for environment installation as explained above.
