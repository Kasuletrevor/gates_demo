import pandas as pd
from typing import List, Any, Dict, Optional

"""
Question answering pipeline.
"""

__all__ = [
    'Search'
]

def __dir__():
    return __all__

class Search:
    """
    Search a vector database to return the most similar context.
    """
    def __init__(
        self, 
        model,
        collection
    ) -> None:
        """
        Initialize the SemanticSearch instance.

        Parameters
        ----------
        model : Model
            An instance of the Model class.
        collection : Collection
            An instance of the Collection class.
        """
        self.model = model
        self.collection = collection        

    def __call__(
        self, 
        question: str,
        answer_dataset,
        n_results: Optional[int] = 3,
        topics: Optional[list] = None,
        distance_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Retrieve similar questions from the DataFrame based on the query results.

        Parameters
        ---------
        question : str
            The search query question text.
        answer_dataset : pd.DataFrame
            The answer dataset.
        n_results : int, default 3
            The number of matched questions to return.
        distance_threshold : float, default None
            The maximum acceptable distance between the search question and the match.

        Returns
        -------
        A list of similar questions.
        """
        input_em = self.model(question)
        results =  self.collection.query(input_em, n_results, topics)
    
        retrieved_ids = results['ids'][0]
        distances = results['distances'][0]
        filtered_ids, filtered_distances, answers, similar_questions = [], [], [], []

        for retrieved_id, distance in zip(retrieved_ids, distances):
            # Filter based on distance_threshold
            if distance_threshold is None or distance <= distance_threshold:
                # Update distances, ids.
                filtered_ids.append(retrieved_id)
                filtered_distances.append(distance)

                # Get the answer to the similar questions.
                answers_to_similar_instance = answer_dataset.query("question_id == @retrieved_id")
                if answers_to_similar_instance.empty:
                    continue

                similar_questions.append(answers_to_similar_instance['question_text'].unique()[0])
                answers.append(list(answers_to_similar_instance['answer_text'][:n_results].values))

        # Prepare filtered results.
        filtered_results = {
            'ids': filtered_ids,
            'distances': filtered_distances,
            'similar_questions': similar_questions,
            'answers': answers
        }

        return filtered_results