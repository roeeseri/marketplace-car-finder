"""Search package exports are intentionally lazy.

Importing the embedding stack pulls in optional third-party ML dependencies,
so callers should import concrete modules directly when they need them.
"""
