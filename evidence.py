# Capture retrieval tool results
# This is a simple in-memory collector. It stores retrieval tool responses for evaluation.


import json


class EvidenceCollector:
    def __init__(self):
        self.chunks = []

    def clear(self):
        self.chunks.clear()

    def add(self, content):
        if isinstance(content, str) and content.strip():
            if content not in self.chunks:
                self.chunks.append(content)

    def get_documents(self):
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

                doc_id = item.get("id")
                content = item.get("content")

                if not doc_id or not isinstance(content, str):
                    continue

                if doc_id not in documents:
                    documents[doc_id] = {
                        "id": doc_id,
                        "title": item.get("title", ""),
                        "content": content,
                    }

        return list(documents.values())

    def get_context(self):
        return [
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
            for doc in self.get_documents()
        ]


evidence_collector = EvidenceCollector()

