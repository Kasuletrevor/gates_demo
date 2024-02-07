"""
Utilities to run the question answering pipelines.
"""

import sys
import pandas as pd
from tqdm import tqdm
from question_answering.search import Search
from question_answering.model import Model
from question_answering.vector_database import VdbClient
from pathlib import Path
from typing import Union

sys.path.append(Path.cwd().parent.parent)
from config import settings

def __dir__():
    return __all__

__all__ = [
    'init_vector_database',
    'extract_question_embeddings',
    'preprocess_update_data'
]


# SET UP CONFIG VARIABLES
model_path = Path(settings.semantic_model_path)
qa_dataset_path = Path(settings.qa_dataset_path)
vector_database_path = Path(settings.vector_database_path)
qa_questions_dataset_name = Path(settings.qa_questions_dataset_name)
qa_answers_dataset_name = Path(settings.qa_answers_dataset_name)
collection_name = settings.collection_name

def ask_a_question(
    question: str
) -> dict:
    """
    Ask a question to get an answer if it exists.
    """
    # Init.
    vector_database_client = VdbClient(str(vector_database_path))
    search_client = Search(
        model=Model(model_path), 
        collection=vector_database_client.get_collection(collection_name)
    )
    answer_dataset = pd.read_csv(qa_dataset_path/qa_answers_dataset_name)

    # Run the search.
    return search_client(question, answer_dataset=answer_dataset, n_results=1)
    

def init_vector_database(
) -> None:
    """
    Initialize a collection in the vector database by adding embeddings data.
    """    
    # Initialize Vector database instance.
    vdb_client = VdbClient(str(vector_database_path))

    collection = vdb_client.get_or_create_collection(collection_name)
    # Read in QA questions data and extract embeddings.
    ids, embeddings = extract_question_embeddings(
        question_data=pd.read_csv(qa_dataset_path/qa_questions_dataset_name),
        model_name_or_path=model_path
    )
    # Add embeddings to database.
    collection.add_embeddings(ids=ids, embeddings=embeddings)


def extract_question_embeddings(
    question_data,
    model_name_or_path: Union[Path, str]
) -> tuple[list[str], list[list[float]]]:
    """
    Extract embeddings from data given a model.

    Parameters
    ----------
    question_data : pd.DataFrame
        A file containing the questions and their unique identifiers.
    model_name_or_path : str
        Name of the SentenceTransformer model or path to model weights.

    Returns
    -------
    list[str], list[list[float]]
        A tuple with a list of ids and embeddings.
    """
    model = Model(model_name_or_path)
    data = question_data[['question_id', 'question_text']]
    ids, embeddings = [], []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        ids.append(row['question_id'])
        embeddings.append(model(row['question_text']))

    return ids, embeddings


def preprocess_update_data(
    update_data
) -> None:
    """
    Process question update data for upsert to answers database and questions for
    the vector database.
    """
    raise NotImplementedError('Process currently undefined.')