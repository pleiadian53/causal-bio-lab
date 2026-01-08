#!/bin/bash
# Regenerate all PDFs with Mermaid diagram support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs/causal_inference"
PDF_DIR="$DOCS_DIR/pdf"

cd "$PROJECT_ROOT"

# Check if mermaid-cli is available
if command -v mmdc &> /dev/null; then
    echo "✓ Found mermaid-cli (mmdc)"
    MERMAID_METHOD="mmdc"
elif python3 -c "import playwright" 2>/dev/null; then
    echo "✓ Found Playwright"
    MERMAID_METHOD="playwright"
else
    echo "⚠️  Warning: No local Mermaid converter found"
    echo "   Install: npm install -g @mermaid-js/mermaid-cli"
    echo "   Or: pip install playwright && playwright install chromium"
    echo "   Will try API method (requires network)..."
    MERMAID_METHOD="api"
fi

# Create PDF directory
mkdir -p "$PDF_DIR"

# Convert each markdown file
echo ""
echo "Converting markdown files to PDF with Mermaid support..."
echo "============================================================"

for md_file in "$DOCS_DIR"/*.md; do
    if [ ! -f "$md_file" ]; then
        continue
    fi
    
    filename=$(basename "$md_file" .md)
    pdf_file="$PDF_DIR/${filename}.pdf"
    
    echo ""
    echo "📄 Processing: $filename"
    echo "   Input:  $md_file"
    echo "   Output: $pdf_file"
    
    if python3 "$SCRIPT_DIR/md_to_pdf.py" "$md_file" -o "$pdf_file" --mermaid-method "$MERMAID_METHOD"; then
        size=$(du -h "$pdf_file" | cut -f1)
        echo "   ✅ Success! Size: $size"
    else
        echo "   ❌ Failed!"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "✅ All PDFs generated successfully!"
echo ""
echo "PDFs are in: $PDF_DIR"
if [ -d "$PDF_DIR/mermaid_images" ]; then
    echo "Mermaid images in: $PDF_DIR/mermaid_images"
fi



