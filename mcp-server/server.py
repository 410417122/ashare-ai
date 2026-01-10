"""
FinLab MCP Server - Provides FinLab documentation access via Model Context Protocol

This server reads documentation from the single source of truth:
finlab-plugin/skills/finlab/*.md
"""

from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
server = FastMCP("finlab-docs")

# Locate the single source of truth for documentation
DOCS_DIR = Path(__file__).parent.parent / "finlab-plugin" / "skills" / "finlab"

# Files to exclude from documentation listing
EXCLUDED_FILES = {"SKILL.md", "README.md"}


def load_doc(name: str) -> str:
    """Load a document from the single source of truth."""
    doc_path = DOCS_DIR / f"{name}.md"
    if not doc_path.exists():
        doc_path = DOCS_DIR / name
        if not doc_path.exists():
            raise FileNotFoundError(f"Document '{name}' not found")
    return doc_path.read_text(encoding="utf-8")


def get_available_docs() -> list[str]:
    """Get list of available document names."""
    return [
        f.stem for f in sorted(DOCS_DIR.glob("*.md"))
        if f.name not in EXCLUDED_FILES
    ]


def search_in_docs(query: str) -> list[dict]:
    """Search for a keyword in all documentation files."""
    results = []
    query_lower = query.lower()

    for md_file in DOCS_DIR.glob("*.md"):
        if md_file.name in EXCLUDED_FILES:
            continue

        content = md_file.read_text(encoding="utf-8")
        if query_lower not in content.lower():
            continue

        # Find matching lines with context
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                # Get context: 2 lines before and 5 lines after
                start = max(0, i - 2)
                end = min(len(lines), i + 6)
                context = "\n".join(lines[start:end])
                results.append({
                    "file": md_file.stem,
                    "line": i + 1,
                    "match": context
                })

    return results[:10]  # Limit results


@server.tool()
def list_documents() -> str:
    """List all available FinLab documentation files.

    Returns a list of document names that can be retrieved with get_document().
    """
    docs = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        if md_file.name in EXCLUDED_FILES:
            continue

        content = md_file.read_text(encoding="utf-8")
        # Extract first heading as title
        first_line = ""
        for line in content.split("\n"):
            if line.strip():
                first_line = line.strip("# ").strip()
                break

        docs.append(f"- **{md_file.stem}**: {first_line}")

    return "## Available FinLab Documents\n\n" + "\n".join(docs)


@server.tool()
def get_document(doc_name: str) -> str:
    """Get the full content of a FinLab documentation file.

    Args:
        doc_name: Name of the document (without .md extension).
                  Available: data-reference, backtesting-reference, dataframe-reference,
                  factor-examples, factor-analysis-reference, trading-reference,
                  best-practices, machine-learning-reference
    """
    try:
        return load_doc(doc_name)
    except FileNotFoundError:
        available = get_available_docs()
        return f"Document '{doc_name}' not found.\n\nAvailable documents:\n" + "\n".join(f"- {d}" for d in available)


@server.tool()
def search_finlab_docs(query: str) -> str:
    """Search for a keyword or phrase in all FinLab documentation.

    Args:
        query: The search term to look for (case-insensitive)
    """
    results = search_in_docs(query)

    if not results:
        return f"No results found for '{query}'"

    output = f"## Search Results: {query}\n\n"
    for r in results:
        output += f"### {r['file']} (line {r['line']})\n"
        output += f"```\n{r['match']}\n```\n\n"

    return output


@server.tool()
def get_factor_examples(factor_type: str = "all") -> str:
    """Get factor/strategy examples from the documentation.

    Args:
        factor_type: Type of factor to filter by. Options:
                     - "all": All examples
                     - "value": Value investing factors (PE, PB, etc.)
                     - "momentum": Price momentum strategies
                     - "technical": Technical analysis indicators
                     - "quality": Quality factors (ROE, margins, etc.)
                     - "ml": Machine learning strategies
    """
    try:
        content = load_doc("factor-examples")
    except FileNotFoundError:
        return "factor-examples.md not found"

    if factor_type == "all":
        return content

    # Search for section matching the factor type
    factor_type_lower = factor_type.lower()
    sections = content.split("\n## ")

    matching_sections = []
    for section in sections:
        if factor_type_lower in section.lower():
            matching_sections.append("## " + section)

    if not matching_sections:
        return f"No examples found for factor type '{factor_type}'. Try: value, momentum, technical, quality, ml"

    return "\n\n".join(matching_sections)
