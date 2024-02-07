"""
Vector database to store embeddings.
"""
import chromadb
from typing import List, Dict, Optional, Any

__all__ = [ 
    'Collection',
    'VdbClient'
]

def __dir__():
    return __all__

class Collection:
    """
    Vector database collection for writing embeddings or documents and querying similarity
    from the vector database.
    """
    def __init__(
        self, 
        collection
    ) -> None:
        self.collection = collection

    def add_documents(
        self, 
        ids: List[str],
        documents: List[str], 
        metadatas: Dict
    ) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        ids : List[str]
            List of IDs for each document.
        documents : List[str]
            List of document strings.
        metadatas : Dict
            Dictionary of metadata for each document.
        """
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def add_embeddings(
        self, 
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add embeddings to the collection.

        Parameters
        ----------
        ids : List[str]
            List of IDs for each embedding.
        embeddings : List[List[float]]
            List of embedding lists.
        metadatas : Optional[List[Dict[str, Any]]], optional
            List of metadata dictionaries for each embedding, by default None.
        """
        self.collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)


    def query(
        self, 
        input_em, 
        n_results: int,
        topics: Optional[list] = None
    ):
        """
        Query the collection using input embeddings.

        Parameters
        ----------
        input_em: ...
        n_results: ...

        Returns
        -------
        Matched documents/queries/terms in collection.
        """
        if topics:
            # Create a list of conditions for the "category" field
            category_conditions = [{"category": topic} for topic in topics]

            # Add a condition for the comma-separated combination
            category_conditions.append({"category": ",".join(topics)})

            # Use the "$or" operator to combine the conditions
            where_clause = {"$or": category_conditions}
            return self.collection.query(query_embeddings=[input_em], n_results=n_results, where=where_clause)
        
        return self.collection.query(query_embeddings=[input_em], n_results=n_results)


class VdbClient:
    """
    Vector database client for creating and deleting
    clients for the vector database.

    Parameters
    ----------
    database_path : str
        The directory in which to store the database.
    """
    def __init__(
        self,
        database_path: str
    ) -> None:
        self.client = chromadb.PersistentClient(path=database_path)

    def get_or_create_collection(
        self, 
        name: str
    ) -> Collection:
        """
        Creates or gets a new collection in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.

        Returns
        -------
        The created collection instance.
        """
        return Collection(self.client.get_or_create_collection(name))
    
    def get_collection(
        self, 
        name: str
    ) -> Collection:
        """
        Gets a collection in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.

        Returns
        -------
        The created collection instance.
        """
        return Collection(self.client.get_collection(name))

    def delete_collection(
        self, 
        name: str
    ) -> None:
        """
        Deletes a collection from the vector database.

        Parameters
        ----------
        name : str
            Name of the collection to delete.
        """
        self.client.delete_collection(name=name)