from uuid import uuid4

from langchain_core.tools import tool


@tool
def create_support_ticket(
    title: str,
    description: str,
) -> dict:
    """Create a support ticket for an issue reported by the user.

    Use this tool only when the user explicitly requests
    creation of a support ticket.
    """
    ticket_id = f"MOCK-{uuid4().hex[:8].upper()}"

    print(f"\n[MOCK TICKET CREATED] {ticket_id}")

    return {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "status": "created",
    }
