#!/usr/bin/env python3
"""
Preprocess Markdown files to convert Mermaid diagrams to images.

This script extracts Mermaid code blocks from markdown files and converts them
to images (SVG/PNG) that can be embedded in PDFs.

Methods supported:
1. mermaid-cli (mmdc) - Recommended, requires: npm install -g @mermaid-js/mermaid-cli
2. Playwright (Python) - Requires: pip install playwright && playwright install chromium
3. Online API fallback - Requires network access

Usage:
    python scripts/preprocess_mermaid.py input.md output.md
    python scripts/preprocess_mermaid.py input.md output.md --method playwright
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib
import base64


class MermaidConverter:
    """Convert Mermaid diagrams to images."""
    
    @staticmethod
    def check_mmdc() -> bool:
        """Check if mermaid-cli (mmdc) is available."""
        try:
            result = subprocess.run(['mmdc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def check_playwright() -> bool:
        """Check if playwright is available."""
        try:
            import playwright
            return True
        except ImportError:
            return False
    
    @staticmethod
    def extract_mermaid_blocks(content: str) -> List[Tuple[int, int, str]]:
        """
        Extract Mermaid code blocks from markdown.
        
        Returns:
            List of (start_pos, end_pos, mermaid_code) tuples
        """
        pattern = r'```mermaid\n(.*?)```'
        matches = []
        for match in re.finditer(pattern, content, re.DOTALL):
            start = match.start()
            end = match.end()
            code = match.group(1).strip()
            matches.append((start, end, code))
        return matches
    
    @staticmethod
    def hash_mermaid(code: str) -> str:
        """Generate hash for Mermaid code (for caching)."""
        return hashlib.md5(code.encode()).hexdigest()[:8]
    
    @staticmethod
    def convert_with_mmdc(mermaid_code: str, output_path: Path, 
                         format: str = 'svg', debug: bool = False) -> bool:
        """Convert Mermaid to image using mermaid-cli."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mermaid_code)
            temp_input = Path(f.name)
        
        try:
            cmd = [
                'mmdc',
                '-i', str(temp_input),
                '-o', str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if debug and result.returncode != 0:
                print(f"\n    mmdc stderr: {result.stderr}", file=sys.stderr)
            return result.returncode == 0 and output_path.exists()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            if debug:
                print(f"\n    mmdc exception: {e}", file=sys.stderr)
            return False
        finally:
            temp_input.unlink(missing_ok=True)
    
    @staticmethod
    def convert_with_playwright(mermaid_code: str, output_path: Path,
                               format: str = 'svg') -> bool:
        """Convert Mermaid to image using Playwright."""
        try:
            from playwright.sync_api import sync_playwright
            
            # Mermaid HTML template
            html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</body>
</html>
"""
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content(html_template)
                page.wait_for_selector('.mermaid svg', timeout=10000)
                
                if format == 'svg':
                    svg_content = page.query_selector('.mermaid svg').inner_html()
                    # Get the full SVG element
                    full_svg = page.evaluate('''() => {
                        const svg = document.querySelector('.mermaid svg');
                        return svg.outerHTML;
                    }''')
                    output_path.write_text(full_svg)
                else:
                    # For PNG, take screenshot
                    svg_element = page.query_selector('.mermaid')
                    svg_element.screenshot(path=str(output_path))
                
                browser.close()
                return True
                
        except Exception as e:
            print(f"Playwright conversion failed: {e}", file=sys.stderr)
            return False
    
    @staticmethod
    def convert_with_api(mermaid_code: str, output_path: Path,
                        format: str = 'svg') -> bool:
        """Convert Mermaid using online API (requires network)."""
        try:
            import urllib.request
            import urllib.parse
            import json
            
            # Encode Mermaid code
            encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()
            
            if format == 'svg':
                url = f"https://mermaid.ink/svg/{encoded}"
            else:
                url = f"https://mermaid.ink/img/{encoded}"
            
            urllib.request.urlretrieve(url, output_path)
            return output_path.exists() and output_path.stat().st_size > 0
            
        except Exception as e:
            print(f"API conversion failed: {e}", file=sys.stderr)
            return False
    
    @staticmethod
    def convert_mermaid(mermaid_code: str, output_path: Path,
                       method: str = 'auto', format: str = 'svg', debug: bool = False) -> bool:
        """
        Convert Mermaid code to image.
        
        Args:
            mermaid_code: Mermaid diagram code
            output_path: Path to save image
            method: 'auto', 'mmdc', 'playwright', or 'api'
            format: 'svg' or 'png'
            debug: Print debug output on failure
        """
        if method == 'auto':
            # Try methods in order of preference
            if MermaidConverter.check_mmdc():
                return MermaidConverter.convert_with_mmdc(mermaid_code, output_path, format, debug)
            elif MermaidConverter.check_playwright():
                return MermaidConverter.convert_with_playwright(mermaid_code, output_path, format)
            else:
                return MermaidConverter.convert_with_api(mermaid_code, output_path, format)
        elif method == 'mmdc':
            return MermaidConverter.convert_with_mmdc(mermaid_code, output_path, format, debug)
        elif method == 'playwright':
            return MermaidConverter.convert_with_playwright(mermaid_code, output_path, format)
        elif method == 'api':
            return MermaidConverter.convert_with_api(mermaid_code, output_path, format)
        else:
            raise ValueError(f"Unknown method: {method}")


def preprocess_markdown(input_path: Path, output_path: Path,
                       method: str = 'auto', image_dir: Optional[Path] = None,
                       format: str = 'svg', debug: bool = False) -> Tuple[int, int]:
    """
    Preprocess markdown file to convert Mermaid diagrams to images.
    
    Returns:
        Tuple of (total_diagrams, successful_conversions)
    """
    content = input_path.read_text()
    mermaid_blocks = MermaidConverter.extract_mermaid_blocks(content)
    
    if not mermaid_blocks:
        # No Mermaid diagrams, just copy the file
        output_path.write_text(content)
        return 0, 0
    
    # Create image directory
    if image_dir is None:
        image_dir = output_path.parent / 'mermaid_images'
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each Mermaid block
    replacements = []
    successful = 0
    
    for i, (start, end, mermaid_code) in enumerate(mermaid_blocks):
        # Generate image filename
        diagram_hash = MermaidConverter.hash_mermaid(mermaid_code)
        image_filename = f"mermaid_{i+1}_{diagram_hash}.{format}"
        image_path = image_dir / image_filename
        
        # Convert Mermaid to image
        print(f"  Converting diagram {i+1}/{len(mermaid_blocks)}...", end=' ')
        if MermaidConverter.convert_mermaid(mermaid_code, image_path, method, format, debug):
            print("✓")
            successful += 1
            # Calculate relative path from output to image
            rel_image_path = image_path.relative_to(output_path.parent)
            # Replace code block with image reference
            image_markdown = f'![Mermaid diagram]({rel_image_path})'
            replacements.append((start, end, image_markdown))
        else:
            print("✗")
            # Keep original code block if conversion fails
            replacements.append((start, end, content[start:end]))
    
    # Apply replacements in reverse order to preserve indices
    replacements.sort(key=lambda x: x[0], reverse=True)
    new_content = content
    for start, end, replacement in replacements:
        new_content = new_content[:start] + replacement + new_content[end:]
    
    output_path.write_text(new_content)
    return len(mermaid_blocks), successful


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess markdown to convert Mermaid diagrams to images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best method
  python scripts/preprocess_mermaid.py input.md output.md
  
  # Use specific method
  python scripts/preprocess_mermaid.py input.md output.md --method mmdc
  
  # Use PNG instead of SVG
  python scripts/preprocess_mermaid.py input.md output.md --format png

Installation:
  # Method 1: mermaid-cli (recommended)
  npm install -g @mermaid-js/mermaid-cli
  
  # Method 2: Playwright (Python)
  pip install playwright
  playwright install chromium
  
  # Method 3: Online API (no installation, requires network)
  # Works automatically if other methods unavailable
        """
    )
    
    parser.add_argument('input', type=Path, help='Input markdown file')
    parser.add_argument('output', type=Path, help='Output markdown file')
    parser.add_argument('--method', type=str, default='auto',
                       choices=['auto', 'mmdc', 'playwright', 'api'],
                       help='Conversion method (default: auto)')
    parser.add_argument('--format', type=str, default='svg',
                       choices=['svg', 'png'],
                       help='Image format (default: svg)')
    parser.add_argument('--image-dir', type=Path,
                       help='Directory to save images (default: output_dir/mermaid_images)')
    parser.add_argument('--debug', action='store_true',
                       help='Print debug output on conversion failures')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    # Check available methods
    if args.method == 'auto':
        if MermaidConverter.check_mmdc():
            print("✓ Using mermaid-cli (mmdc)")
        elif MermaidConverter.check_playwright():
            print("✓ Using Playwright")
        else:
            print("⚠️  No local converters found, will use online API (requires network)")
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Preprocess
    print(f"📖 Processing: {args.input}")
    total, successful = preprocess_markdown(
        args.input, args.output, args.method, args.image_dir, args.format, args.debug
    )
    
    if total > 0:
        print(f"✅ Converted {successful}/{total} Mermaid diagrams")
        print(f"📄 Output: {args.output}")
        if successful < total:
            print(f"⚠️  {total - successful} diagrams failed to convert", file=sys.stderr)
    else:
        print("ℹ️  No Mermaid diagrams found")


if __name__ == '__main__':
    main()

