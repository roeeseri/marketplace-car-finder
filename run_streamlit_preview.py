from __future__ import annotations

import sys

import streamlit.web.cli as cli


def main() -> None:
    sys.argv = [
        "streamlit",
        "run",
        "streamlit_mockup.py",
        "--global.developmentMode=false",
        "--server.headless=true",
        "--server.port=8503",
        "--browser.gatherUsageStats=false",
    ]
    cli.main()


if __name__ == "__main__":
    main()
