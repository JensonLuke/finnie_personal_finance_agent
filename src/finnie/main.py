"""CLI entry point (`finnie` console script).

The primary UI is the Streamlit app (`streamlit run src/finnie/web_app/app.py`,
per docs/design/langgraph-workflow.md and the Milestones doc's UI plan).
This entry point is a thin convenience wrapper, not the main way the app runs.
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "Finnie is a Streamlit app. Run it with:\n\n"
        "    streamlit run src/finnie/web_app/app.py\n\n"
        "To (re)build the RAG index first, run:\n\n"
        "    python -m finnie.rag.build_index\n"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
