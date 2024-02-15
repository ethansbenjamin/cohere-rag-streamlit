import hnswlib
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from cohere.responses.chat import StreamEvent


class Documents:
    """
    A class representing a collection of documents.

    Parameters:
    sources (list): A list of dictionaries representing the sources of the documents. Each dictionary should have 'title' and 'url' keys.

    Attributes:
    sources (list): A list of dictionaries representing the sources of the documents.
    docs (list): A list of dictionaries representing the documents, with 'title', 'content', and 'url' keys.
    docs_embs (list): A list of the associated embeddings for the documents.
    retrieve_top_k (int): The number of documents to retrieve during search.
    rerank_top_k (int): The number of documents to rerank after retrieval.
    docs_len (int): The number of documents in the collection.
    index (hnswlib.Index): The index used for document retrieval.

    Methods:
    load(): Loads the data from the sources and partitions the HTML content into chunks.
    embed(): Embeds the documents using the Cohere API.
    index(): Indexes the documents for efficient retrieval.
    retrieve(query): Retrieves documents based on the given query.

    """

    def __init__(self, sources: List[Dict[str, str]], client):
        self.sources = sources
        self.co = client
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load()
        self.embed()
        self.index()

    def load(self) -> None:
        """
        Loads the documents from the sources and chunks the HTML content.
        """
        print("Loading documents...")

        for source in self.sources:
            elements = partition_html(url=source["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                self.docs.append(
                    {
                        "title": source["title"],
                        "text": str(chunk),
                        "url": source["url"],
                    }
                )

    def embed(self) -> None:
        """
        Embeds the documents using the Cohere API.
        """
        print("Embedding documents...")

        batch_size = 90
        self.docs_len = len(self.docs)

        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = self.co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the documents for efficient retrieval.
        """
        print("Indexing documents...")

        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len,
                            ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

        print(
            f"Indexing complete with {self.idx.get_current_count()} documents.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the given query.

        Parameters:
        query (str): The query to retrieve documents for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents, with 'title', 'text', and 'url' keys.
        """
        docs_retrieved = []
        query_emb = self.co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        docs_to_rerank = []
        for doc_id in doc_ids:
            docs_to_rerank.append(self.docs[doc_id]["text"])

        rerank_results = self.co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v2.0",
        )

        doc_ids_reranked = []
        for result in rerank_results:
            doc_ids_reranked.append(doc_ids[result.index])

        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved


class Chatbot:
    """
    A class representing a chatbot.

    Parameters:
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Attributes:
    conversation_id (str): The unique ID for the conversation.
    docs (Documents): An instance of the Documents class representing the collection of documents.

    Methods:
    generate_response(message): Generates a response to the user's message.
    retrieve_docs(response): Retrieves documents based on the search queries in the response.

    """

    def __init__(self, docs: Documents, client):
        self.docs = docs
        self.co = client
        self.conversation_id = str(uuid.uuid4())

    def generate_response(self, message: str):
        """
        Generates a response to the user's message.

        Parameters:
        message (str): The user's message.

        Yields:
        Event: A response event generated by the chatbot.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Generate search queries (if any)
        response = self.co.chat(message=message, search_queries_only=True)

        # If there are search queries, retrieve documents and respond
        # preamble_override = "You only answer questions using on the documents you have provided with"

        if response.search_queries:
            print("Retrieving information...")

            documents = self.retrieve_docs(response)

            response = self.co.chat(
                message=message,
                # preamble_override = preamble_override,
                documents=documents,
                conversation_id=self.conversation_id,
                stream=True,
            )
            for event in response:
                if (event.event_type == StreamEvent.TEXT_GENERATION):
                    yield event.text

        # If there is no search query, directly respond
        else:
            response = self.co.chat(
                message=message,
                # preamble_override = preamble_override,
                conversation_id=self.conversation_id,
                stream=True
            )

            for event in response:
                if (event.event_type == StreamEvent.TEXT_GENERATION):
                    yield event.text

    def retrieve_docs(self, response) -> List[Dict[str, str]]:
        """
        Retrieves documents based on the search queries in the response.

        Parameters:
        response: The response object containing search queries.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved documents.

        """
        # Get the query(s)
        queries = []
        for search_query in response.search_queries:
            queries.append(search_query["text"])

        # Retrieve documents for each query
        retrieved_docs = []
        for query in queries:
            retrieved_docs.extend(self.docs.retrieve(query))

        return retrieved_docs
