from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Union, Optional

__all__ = [
    'Model'
]

def __dir__():
    return __all__

class Model:
    """
    Embeddings extraction using Sentence Transformers.
    """
    def __init__(self, 
        model_name_or_path: Union[str, Path],
        max_seq_length: Optional[int] = 80
    ) -> None:
        """
        Initializes the Model instance.

        Parameters
        ----------
        model_name_or_path : str
            Name of the SentenceTransformer model or path to model weights.
        max_seq_length : int, default 80
            Maximum sequence length for encoding.
        """
        self.model = SentenceTransformer(model_name_or_path)
        self.model.max_seq_length = max_seq_length
            
    def __call__(
        self, 
        question_text: str
    ) -> list[float]:
        """
        Encodes a question into a list of embeddings.

        Parameters
        ----------
        question_text : str
            The input question.

        Returns
        --------
        A list of embeddings.
        """
        return self.model.encode(question_text).tolist()