# Capture retrieval tool results
# This is a simple in-memory collector. It stores retrieval tool responses for evaluation.


import json
class EvidenceCollector:
    """Parse and deduplicate retrieved chunks by unique chunk ID."""

    def __init__(self):
        self.chunks: list[str] = []

    def add(self, content):
        if isinstance(content, str) and content.strip():
            if content not in self.chunks:
                self.chunks.append(content)

    def get_documents(self) -> list[dict]:
        documents = {}

        for response in self.chunks:
            try:
                items = json.loads(response)
            except json.JSONDecodeError:
                continue

            if not isinstance(items, list):
                continue

            for item in items:
                if not isinstance(item, dict):
                    continue

                chunk_id = item.get("chunk_id") or item.get("id")
                content = item.get("content")

                if (
                    not isinstance(chunk_id, str)
                    or not chunk_id
                    or not isinstance(content, str)
                    or not content.strip()
                ):
                    continue

                if chunk_id not in documents:
                    documents[chunk_id] = {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "document_id": item.get("document_id"),
                        "title": item.get("title", ""),
                        "content": content,
                        "source": item.get("source", ""),
                        "section": item.get("section", {}),
                    }

        return list(documents.values())

    def get_context(self) -> list[str]:
        return [
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
            for doc in self.get_documents()
        ]
