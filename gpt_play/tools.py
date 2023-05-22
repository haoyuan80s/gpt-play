from sympy import sympify


def calculator(query: str) -> str:
    """Useful for when you need to answer questions about math."""
    try:
        result = sympify(query)
        return str(result)
    except (SyntaxError, TypeError):
        return "Invalid input."
