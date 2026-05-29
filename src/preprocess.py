import re

INTENT_MAP = {
    0: "Refund Request",
    1: "Bug Report",
    2: "Feature Request",
    3: "Delivery Complaint",
    4: "Product Inquiry",
    5: "Cancellation",
    6: "Praise",
    7: "Technical Support"
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def assign_intent(text):

    text = text.lower()

    if any(word in text for word in [
        "refund",
        "return",
        "money back",
        "replacement"
    ]):
        return 0

    elif any(word in text for word in [
        "bug",
        "error",
        "crash",
        "issue",
        "problem",
        "broken"
    ]):
        return 1

    elif any(word in text for word in [
        "feature",
        "suggestion",
        "improve",
        "add"
    ]):
        return 2

    elif any(word in text for word in [
        "delivery",
        "shipment",
        "shipping",
        "late",
        "package"
    ]):
        return 3

    elif any(word in text for word in [
        "how",
        "what",
        "when",
        "where",
        "can i"
    ]):
        return 4

    elif any(word in text for word in [
        "cancel",
        "unsubscribe"
    ]):
        return 5

    elif any(word in text for word in [
        "excellent",
        "amazing",
        "great",
        "awesome",
        "love"
    ]):
        return 6

    else:
        return 7