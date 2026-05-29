def route_ticket(intent):

    routing = {
        "Refund Request": "Customer Support Team",
        "Bug Report": "Engineering Team",
        "Feature Request": "Product Team",
        "Delivery Complaint": "Logistics Team",
        "Product Inquiry": "Sales Team",
        "Cancellation": "Billing Team",
        "Praise": "Customer Success Team",
        "Technical Support": "Technical Support Team"
    }

    return routing.get(
        intent,
        "General Support"
    )


def detect_priority(text):

    text = text.lower()

    urgent_words = [
        "urgent",
        "immediately",
        "asap",
        "critical",
        "emergency"
    ]

    for word in urgent_words:
        if word in text:
            return "High"

    return "Normal"