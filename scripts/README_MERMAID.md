# Mermaid Diagram Support in PDF Conversion

The PDF conversion scripts now support Mermaid diagrams! Mermaid code blocks in markdown files are automatically converted to images before PDF generation.

## Quick Start

### Option 1: Install mermaid-cli (Recommended)

```bash
# Install mermaid-cli globally
npm install -g @mermaid-js/mermaid-cli

# Now PDF conversion will automatically handle Mermaid diagrams
python3 scripts/md_to_pdf.py docs/causal_inference/backdoor_paths.md -o docs/causal_inference/pdf/backdoor_paths.pdf
```

### Option 2: Use Playwright (Python)

```bash
# Install Playwright
pip install playwright
playwright install chromium

# Use Playwright method
python3 scripts/md_to_pdf.py docs/causal_inference/backdoor_paths.md \
    --mermaid-method playwright
```

### Option 3: Use Online API (Requires Network)

```bash
# Uses mermaid.ink API (requires internet connection)
python3 scripts/md_to_pdf.py docs/causal_inference/backdoor_paths.md \
    --mermaid-method api
```

## How It Works

1. **Preprocessing**: The script detects Mermaid code blocks (```` ```mermaid ````)
2. **Conversion**: Each diagram is converted to an SVG/PNG image
3. **Replacement**: Code blocks are replaced with image references
4. **PDF Generation**: Pandoc converts the modified markdown to PDF

## Manual Preprocessing

You can also preprocess Mermaid diagrams separately:

```bash
# Preprocess a single file
python3 scripts/preprocess_mermaid.py input.md output.md

# Use specific method
python3 scripts/preprocess_mermaid.py input.md output.md --method mmdc

# Use PNG instead of SVG
python3 scripts/preprocess_mermaid.py input.md output.md --format png
```

## Batch Conversion

To regenerate all PDFs with Mermaid support:

```bash
# Make sure mermaid-cli is installed first
npm install -g @mermaid-js/mermaid-cli

# Then run batch conversion
cd /Users/pleiadian53/work/causal-bio-lab
for f in docs/causal_inference/*.md; do
  name=$(basename "$f" .md)
  echo "Converting: $f"
  python3 scripts/md_to_pdf.py "$f" -o "docs/causal_inference/pdf/${name}.pdf"
  echo ""
done
```

## Troubleshooting

### "Mermaid CLI not found"
- Install: `npm install -g @mermaid-js/mermaid-cli`
- Verify: `mmdc --version`

### "Playwright not available"
- Install: `pip install playwright && playwright install chromium`

### Diagrams not rendering in PDF
- Check that images were created in `docs/causal_inference/pdf/mermaid_images/`
- Try using `--mermaid-method mmdc` explicitly
- Check the preprocessed markdown: `python3 scripts/preprocess_mermaid.py input.md test_output.md`

### Skip Mermaid preprocessing
```bash
python3 scripts/md_to_pdf.py input.md --no-mermaid
```

## File Structure

After conversion, you'll have:
```
docs/causal_inference/
├── backdoor_paths.md          # Original markdown
├── pdf/
│   ├── backdoor_paths.pdf     # Generated PDF
│   └── mermaid_images/        # Generated diagram images
│       ├── mermaid_1_abc123.svg
│       └── mermaid_2_def456.svg
```

## Methods Comparison

| Method | Pros | Cons | Installation |
|--------|------|------|--------------|
| **mmdc** | Fast, offline, best quality | Requires npm | `npm install -g @mermaid-js/mermaid-cli` |
| **Playwright** | Works offline, flexible | Slower, larger install | `pip install playwright` |
| **API** | No installation | Requires network | None |

**Recommendation**: Use `mmdc` (mermaid-cli) for best results.



