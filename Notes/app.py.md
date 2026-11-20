## **Topic 1: Dictionary unpacking**

In Deal(**item), ** means dictionary unpacking. It takes the key–value pairs from the dictionary item and passes them to Deal as named arguments.

Suppose item contains:
item = { "product_description": "Wireless keyboard", "price": 25.99, "url": "https://example.com/keyboard"}

When Python runs:
deal = Deal(\*\*item)

It is equivalent to writing:
deal = Deal( product_description="Wireless keyboard", price=25.99, url="https://example.com/keyboard")

The dictionary keys become argument names, and the dictionary values become their values. Your Deal model then checks those values and creates the object.
