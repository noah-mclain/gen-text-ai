#!/usr/bin/env python3
"""
Prompt Enhancer

This script enhances user prompts for code and text by:
1. Fixing spelling mistakes
2. Correcting spacing issues
3. Enhancing clarity by adding missing words (using language models)
4. Formatting code correctly
5. Providing a side-by-side comparison of original and enhanced prompts

Usage:
    python -m scripts.utilities.prompt_enhancer --prompt "Your prompt text" [OPTIONS]
    
    OR
    
    python -m scripts.utilities.prompt_enhancer --file prompt.txt [OPTIONS]
"""

import os
import re
import argparse
import logging
import sys
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import difflib

# Import NLP libraries
from textblob import TextBlob
import spacy
import nltk

# Try importing syntax highlighters
try:
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import TerminalFormatter
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Define a simple sentence tokenizer as backup
def simple_sentence_tokenize(text):
    """Simple sentence tokenizer that splits on periods, exclamation marks, and question marks."""
    if not text:
        return []
    
    # Handle common abbreviations to avoid splitting them
    text = re.sub(r'(\b(?:Mr|Mrs|Dr|Ms|Prof|Inc|Ltd|Jr|Sr|Co|etc|vs|e\.g|i\.e)\.)(\s)', r'\1TEMPMARKER\2', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore abbreviations
    sentences = [s.replace('TEMPMARKER', '.') for s in sentences]
    
    # Remove empty sentences
    sentences = [s for s in sentences if s.strip()]
    
    return sentences

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model not found. Downloading en_core_web_sm...")
    os.system("python -m spacy download en_core_web_sm")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.error("Failed to load spaCy model. Some features may not work.")
        nlp = None

class PromptEnhancer:
    """Class for enhancing text and code prompts."""
    
    def __init__(self, aggressive: bool = False, fix_code: bool = True, 
                 add_missing_words: bool = True):
        """
        Initialize the PromptEnhancer.
        
        Args:
            aggressive: Whether to be more aggressive in fixing issues
            fix_code: Whether to format and correct code blocks
            add_missing_words: Whether to try to add missing words
        """
        self.aggressive = aggressive
        self.fix_code = fix_code
        self.add_missing_words = add_missing_words
        
        # Code block pattern
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        
    def enhance_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Enhance a prompt by fixing various issues.
        
        Args:
            prompt: The text prompt to enhance
            
        Returns:
            Dictionary with original and enhanced prompts and statistics
        """
        if not prompt or not isinstance(prompt, str):
            return {
                "original": "",
                "enhanced": "",
                "changes": 0,
                "details": {}
            }
        
        # Track changes
        changes = {
            "spelling": 0,
            "spacing": 0,
            "missing_words": 0,
            "code_formatting": 0,
            "punctuation": 0
        }
        
        # Extract code blocks to process them separately
        code_blocks = self.code_block_pattern.findall(prompt)
        
        # Replace code blocks with placeholders to preserve them from text processing
        prompt_with_placeholders = prompt
        for i, (lang, code) in enumerate(code_blocks):
            placeholder = f"CODE_BLOCK_{i}_LANG_{lang}"
            prompt_with_placeholders = prompt_with_placeholders.replace(f"```{lang}\n{code}\n```", placeholder)
        
        # Extract inline code
        inline_codes = self.inline_code_pattern.findall(prompt_with_placeholders)
        
        # Replace inline code with placeholders
        for i, code in enumerate(inline_codes):
            placeholder = f"INLINE_CODE_{i}"
            prompt_with_placeholders = prompt_with_placeholders.replace(f"`{code}`", placeholder)
        
        # Enhance text
        enhanced_text, text_changes = self._enhance_text(prompt_with_placeholders)
        changes.update(text_changes)
        
        # Process code blocks
        processed_code_blocks = []
        for i, (lang, code) in enumerate(code_blocks):
            if self.fix_code:
                enhanced_code, code_changes = self._enhance_code(code, lang)
                changes["code_formatting"] += code_changes
            else:
                enhanced_code = code
            processed_code_blocks.append((lang, enhanced_code))
        
        # Restore code blocks
        for i, (lang, code) in enumerate(processed_code_blocks):
            placeholder = f"CODE_BLOCK_{i}_LANG_{lang}"
            enhanced_text = enhanced_text.replace(placeholder, f"```{lang}\n{code}\n```")
        
        # Restore inline code
        for i, code in enumerate(inline_codes):
            placeholder = f"INLINE_CODE_{i}"
            if self.fix_code:
                enhanced_code, _ = self._enhance_code(code, "")
            else:
                enhanced_code = code
            enhanced_text = enhanced_text.replace(placeholder, f"`{enhanced_code}`")
        
        # Calculate total changes
        total_changes = sum(changes.values())
        
        return {
            "original": prompt,
            "enhanced": enhanced_text,
            "changes": total_changes,
            "details": changes
        }
    
    def _enhance_text(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Enhance text by fixing spelling, spacing, and adding missing words.
        
        Args:
            text: Text to enhance
            
        Returns:
            Tuple of enhanced text and changes dictionary
        """
        changes = {
            "spelling": 0,
            "spacing": 0,
            "missing_words": 0,
            "punctuation": 0
        }
        
        # Fix spacing issues
        text, spacing_changes = self._fix_spacing(text)
        changes["spacing"] = spacing_changes
        
        # Split into sentences for better processing
        sentences = simple_sentence_tokenize(text)
        enhanced_sentences = []
        
        for sentence in sentences:
            # Fix spelling
            corrected = self._fix_spelling(sentence)
            if corrected != sentence:
                changes["spelling"] += 1
            
            # Fix punctuation
            corrected, punct_changes = self._fix_punctuation(corrected)
            changes["punctuation"] += punct_changes
            
            # Add missing words using spaCy if available
            if self.add_missing_words and nlp:
                corrected, word_changes = self._add_missing_words(corrected)
                changes["missing_words"] += word_changes
            
            enhanced_sentences.append(corrected)
        
        # Join enhanced sentences
        enhanced_text = " ".join(enhanced_sentences)
        
        # Final spacing fixes
        enhanced_text, additional_spacing = self._fix_spacing(enhanced_text)
        changes["spacing"] += additional_spacing
        
        return enhanced_text, changes
    
    def _fix_spelling(self, text: str) -> str:
        """
        Fix spelling mistakes using TextBlob.
        
        Args:
            text: Text to fix
            
        Returns:
            Text with corrected spelling
        """
        try:
            # Skip URLs and technical terms when fixing spelling
            parts = []
            url_pattern = re.compile(r'https?://\S+')
            tech_pattern = re.compile(r'\b(?:[A-Z][a-z]*){2,}\b|\b[a-z]+[A-Z][a-z]*\b|\b[A-Z]{2,}\b')
            
            # Find positions of URLs and technical terms
            urls = list(url_pattern.finditer(text))
            tech_terms = list(tech_pattern.finditer(text))
            all_matches = sorted(urls + tech_terms, key=lambda x: x.start())
            
            # Split text into parts to skip URLs and technical terms
            last_pos = 0
            for match in all_matches:
                if match.start() > last_pos:
                    part = text[last_pos:match.start()]
                    parts.append((part, True))  # (text, should_correct)
                parts.append((match.group(), False))  # Don't correct this part
                last_pos = match.end()
            
            if last_pos < len(text):
                parts.append((text[last_pos:], True))
            
            # Process each part
            corrected_parts = []
            for part, should_correct in parts:
                if should_correct and len(part.split()) > 0:
                    # Only apply spelling correction to regular text
                    blob = TextBlob(part)
                    corrected_parts.append(str(blob.correct()))
                else:
                    corrected_parts.append(part)
            
            return "".join(corrected_parts)
        except Exception as e:
            logger.warning(f"Error fixing spelling: {e}")
            return text
    
    def _fix_spacing(self, text: str) -> Tuple[str, int]:
        """
        Fix spacing issues in text.
        
        Args:
            text: Text to fix
            
        Returns:
            Tuple of fixed text and number of changes
        """
        original = text
        changes = 0
        
        # Fix double spaces
        while "  " in text:
            text = text.replace("  ", " ")
            changes += 1
        
        # Fix spacing around punctuation
        for punct in ".,:;!?)]}>":
            text = re.sub(r'\s+' + re.escape(punct), punct, text)
            changes += len(original) - len(text)
            original = text
        
        for punct in "([{<":
            text = re.sub(re.escape(punct) + r'\s+', punct, text)
            changes += len(original) - len(text)
            original = text
        
        # Ensure space after punctuation if followed by letter
        for punct in ".,:;!?":
            text = re.sub(re.escape(punct) + r'([A-Za-z])', punct + ' ' + r'\1', text)
            
        # Add space after closing punctuation if followed by opening punctuation
        text = re.sub(r'([.,:;!?)])([A-Za-z(])', r'\1 \2', text)
        
        changes += abs(len(original) - len(text))
        
        return text, changes
    
    def _fix_punctuation(self, text: str) -> Tuple[str, int]:
        """
        Fix punctuation issues.
        
        Args:
            text: Text to fix
            
        Returns:
            Tuple of fixed text and number of changes
        """
        original = text
        changes = 0
        
        # Ensure sentence ends with period if it doesn't have ending punctuation
        if re.match(r'^[A-Z]', text) and not re.search(r'[.!?:]$', text):
            text = text + "."
            changes += 1
        
        # Balance quotes and parentheses
        for pair in [("'", "'"), ('"', '"'), ('(', ')'), ('[', ']'), ('{', '}')]:
            open_char, close_char = pair
            count_open = text.count(open_char)
            count_close = text.count(close_char)
            
            if count_open > count_close:
                text = text + close_char * (count_open - count_close)
                changes += count_open - count_close
            elif count_close > count_open and (open_char != close_char):
                text = open_char * (count_close - count_open) + text
                changes += count_close - count_open
        
        return text, changes
    
    def _add_missing_words(self, text: str) -> Tuple[str, int]:
        """
        Add missing words using spaCy's language model.
        
        Args:
            text: Text to enhance
            
        Returns:
            Tuple of enhanced text and number of changes
        """
        if not nlp or len(text.strip()) == 0:
            return text, 0
        
        changes = 0
        doc = nlp(text)
        enhanced_tokens = []
        
        for i, token in enumerate(doc):
            enhanced_tokens.append(token.text)
            
            # Check for missing articles
            if (i < len(doc) - 1 and
                token.pos_ in ("VERB", "ADP", "DET") and
                doc[i+1].pos_ == "NOUN" and
                not doc[i+1].is_stop and
                not any(t.pos_ == "DET" for t in doc[max(0, i-1):i+1])):
                
                # Add missing article ("a" or "an")
                next_word = doc[i+1].text
                if next_word[0].lower() in "aeiou":
                    enhanced_tokens.append("an")
                else:
                    enhanced_tokens.append("a")
                changes += 1
            
            # Check for missing prepositions
            elif (i < len(doc) - 1 and
                  token.pos_ == "VERB" and
                  doc[i+1].pos_ == "NOUN" and
                  token.lemma_ in ("go", "arrive", "come", "travel", "move")):
                
                # Add missing "to" preposition
                enhanced_tokens.append("to")
                changes += 1
        
        enhanced_text = " ".join(enhanced_tokens)
        # Fix spacing issues again
        enhanced_text, _ = self._fix_spacing(enhanced_text)
        
        return enhanced_text, changes
    
    def _enhance_code(self, code: str, lang: str) -> Tuple[str, int]:
        """
        Enhance code by fixing indentation and formatting.
        
        Args:
            code: Code to enhance
            lang: Programming language
            
        Returns:
            Tuple of enhanced code and number of changes
        """
        if not code or not self.fix_code:
            return code, 0
        
        changes = 0
        enhanced_code = code
        
        # Try to guess language if not provided
        if not lang and HAS_PYGMENTS:
            try:
                lexer = guess_lexer(code)
                lang = lexer.name.lower()
            except:
                # Default to Python for formatting
                lang = "python"
        
        # Fix indentation
        if lang.lower() in ("python", "py", ""):
            enhanced_code, indent_changes = self._fix_python_indentation(code)
            changes += indent_changes
        
        # Fix common syntax errors based on language
        if lang.lower() in ("python", "py", ""):
            enhanced_code, syntax_changes = self._fix_python_syntax(enhanced_code)
            changes += syntax_changes
        elif lang.lower() in ("javascript", "js", "typescript", "ts"):
            enhanced_code, syntax_changes = self._fix_js_syntax(enhanced_code)
            changes += syntax_changes
        
        return enhanced_code, changes
    
    def _fix_python_indentation(self, code: str) -> Tuple[str, int]:
        """
        Fix Python code indentation.
        
        Args:
            code: Python code
            
        Returns:
            Tuple of fixed code and number of changes
        """
        lines = code.split('\n')
        if not lines:
            return code, 0
        
        changes = 0
        indent_level = 0
        fixed_lines = []
        block_started = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            orig_leading_spaces = len(line) - len(line.lstrip())
            
            # Skip empty lines
            if not stripped:
                fixed_lines.append("")
                continue
            
            # Decrease indent level for lines with only closing brackets
            if stripped in (')', '}', ']'):
                indent_level = max(0, indent_level - 1)
            
            # Determine if this line should maintain same indentation as previous
            # lines like 'else:', 'elif', 'except:', etc.
            is_continuation = False
            if stripped.startswith(('else:', 'elif ', 'except:', 'except ', 'finally:')):
                # Look for previous non-empty line
                prev_idx = i - 1
                while prev_idx >= 0 and not lines[prev_idx].strip():
                    prev_idx -= 1
                
                if prev_idx >= 0:
                    # If preceded by a line ending with colon, keep same indent
                    prev_line = lines[prev_idx].strip()
                    if prev_line.endswith(':'):
                        is_continuation = True
                        # Reduce indent level for these continuation lines
                        indent_level = max(0, indent_level - 1)
            
            # Apply indentation
            if is_continuation and fixed_lines:
                # Use same indentation as previous block starter
                current_indent = len(fixed_lines[-1]) - len(fixed_lines[-1].lstrip())
                new_line = ' ' * current_indent + stripped
            else:
                # Apply the calculated indent level
                new_line = ' ' * (4 * indent_level) + stripped
            
            # Check if we fixed the indentation
            if new_line != line:
                changes += 1
            
            fixed_lines.append(new_line)
            
            # Increase indent for next line after block start
            if stripped.endswith(':'):
                indent_level += 1
                block_started = True
            else:
                block_started = False
            
            # Handle brackets affecting indent
            open_brackets = stripped.count('(') + stripped.count('{') + stripped.count('[')
            close_brackets = stripped.count(')') + stripped.count('}') + stripped.count(']')
            
            # Only adjust indent level if there's a bracket imbalance
            if open_brackets > close_brackets:
                indent_level += open_brackets - close_brackets
            elif close_brackets > open_brackets and not stripped in (')', '}', ']'):
                # Already handled simple closing bracket lines above
                indent_level = max(0, indent_level - (close_brackets - open_brackets))
        
        return '\n'.join(fixed_lines), changes
    
    def _fix_python_syntax(self, code: str) -> Tuple[str, int]:
        """
        Fix common Python syntax errors.
        
        Args:
            code: Python code
            
        Returns:
            Tuple of fixed code and number of changes
        """
        original = code
        changes = 0
        
        # Fix missing colons after if, for, while, etc.
        pattern = r'(\b(?:if|for|while|def|class|with|try|except|finally|elif|else)\s+[^:]+?)(\s*)$'
        
        def add_missing_colons(match):
            nonlocal changes
            statement = match.group(1)
            whitespace = match.group(2)
            # Only add colon if it's missing
            if not statement.rstrip().endswith(':'):
                changes += 1
                return f"{statement}:{whitespace}"
            return f"{statement}{whitespace}"
        
        # Apply fixes line by line
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                fixed_lines.append(line)
                continue
                
            # Fix missing colons in control statements
            line = re.sub(pattern, add_missing_colons, line)
            
            # Fix missing parentheses in function calls
            if 'print' in line and not re.search(r'print\s*\(', line):
                line = re.sub(r'print\s+(.+)', r'print(\1)', line)
                changes += 1
            
            # Fix missing colons in special cases (if/for/while at end of line)
            if re.search(r'\b(if|for|while|def|class)\b.*\S+\s*$', line) and not line.rstrip().endswith(':'):
                line = line.rstrip() + ':'
                changes += 1
            
            # Fix common Python method name errors
            line = re.sub(r'\.tolowercase\(\)', '.lower()', line)
            line = re.sub(r'\.touppercase\(\)', '.upper()', line)
            
            # Count quotes
            single_quotes = line.count("'")
            double_quotes = line.count('"')
            
            # Fix if odd number of quotes (unclosed)
            if single_quotes % 2 == 1 and "'" in line:
                last_quote_pos = line.rindex("'")
                line = line[:last_quote_pos+1] + "'" + line[last_quote_pos+1:]
                changes += 1
            
            if double_quotes % 2 == 1 and '"' in line:
                last_quote_pos = line.rindex('"')
                line = line[:last_quote_pos+1] + '"' + line[last_quote_pos+1:]
                changes += 1
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        return code, changes
    
    def _fix_js_syntax(self, code: str) -> Tuple[str, int]:
        """
        Fix common JavaScript/TypeScript syntax errors.
        
        Args:
            code: JavaScript/TypeScript code
            
        Returns:
            Tuple of fixed code and number of changes
        """
        original = code
        changes = 0
        
        # Fix missing semicolons
        lines = code.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines, lines that already have semicolons, and block statements
            if (not stripped or stripped.endswith(';') or stripped.endswith('{') or
                stripped.endswith('}') or stripped.endswith(':') or
                stripped.startswith('//')):
                fixed_lines.append(line)
                continue
            
            # Add semicolon to statements that should have them
            if (re.match(r'^(let|const|var|return|throw|break|continue|import|export)', stripped) or
                re.search(r'[a-zA-Z0-9_"\')\]]\s*$', stripped)):
                line = line + ';'
                changes += 1
            
            fixed_lines.append(line)
        
        code = '\n'.join(fixed_lines)
        
        # Fix missing braces in if/for/while statements
        def add_missing_braces(match):
            nonlocal changes
            keyword = match.group(1)
            condition = match.group(2)
            statement = match.group(3)
            changes += 1
            return f"{keyword} {condition} {{\n    {statement}\n}}"
        
        code = re.sub(r'\b(if|for|while)\s+([^{]+)\s+([^{}\n]+);', 
                      add_missing_braces, code)
        
        return code, changes


def display_diff(original: str, enhanced: str) -> str:
    """
    Display a colored diff between original and enhanced text.
    
    Args:
        original: Original text
        enhanced: Enhanced text
        
    Returns:
        Formatted diff as string
    """
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        enhanced.splitlines(keepends=True),
        fromfile='Original',
        tofile='Enhanced',
        n=3
    )
    
    return ''.join(diff)


def display_side_by_side(original: str, enhanced: str, width: int = 80) -> str:
    """
    Display original and enhanced text side by side.
    
    Args:
        original: Original text
        enhanced: Enhanced text
        width: Width for each column
        
    Returns:
        Formatted side-by-side comparison
    """
    # Identify code blocks to preserve them in display
    code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    
    # Replace code blocks with placeholders for display
    def replace_with_placeholders(text, prefix="ORIG"):
        blocks = []
        def replace_block(match):
            lang, code = match.groups()
            placeholder = f"{prefix}_CODE_{len(blocks)}_{lang}"
            blocks.append((lang, code))
            return placeholder
        processed = code_block_pattern.sub(replace_block, text)
        return processed, blocks
    
    # Process both texts
    orig_processed, orig_blocks = replace_with_placeholders(original, "ORIG")
    enh_processed, enh_blocks = replace_with_placeholders(enhanced, "ENH")
    
    # Split into lines
    orig_lines = orig_processed.splitlines()
    enh_lines = enh_processed.splitlines()
    
    # Fill shorter list with empty strings to match length
    max_lines = max(len(orig_lines), len(enh_lines))
    orig_lines.extend([''] * (max_lines - len(orig_lines)))
    enh_lines.extend([''] * (max_lines - len(enh_lines)))
    
    # Calculate column width
    col_width = width // 2
    
    # Format header
    result = []
    result.append("=" * (col_width * 2 + 3))
    result.append(f"{'ORIGINAL':^{col_width}} | {'ENHANCED':^{col_width}}")
    result.append("=" * (col_width * 2 + 3))
    
    # Format each line
    for orig, enh in zip(orig_lines, enh_lines):
        # Process original line
        if orig.startswith("ORIG_CODE_"):
            parts = orig.split("_")
            if len(parts) > 2:
                idx = int(parts[2])
                if idx < len(orig_blocks):
                    lang, code = orig_blocks[idx]
                    orig = f"```{lang}"  # Just show the code block marker
        
        # Process enhanced line
        if enh.startswith("ENH_CODE_"):
            parts = enh.split("_")
            if len(parts) > 2:
                idx = int(parts[2])
                if idx < len(enh_blocks):
                    lang, code = enh_blocks[idx]
                    enh = f"```{lang}"  # Just show the code block marker
        
        # Truncate if too long
        orig = (orig[:col_width-3] + '...') if len(orig) > col_width else orig
        enh = (enh[:col_width-3] + '...') if len(enh) > col_width else enh
        
        result.append(f"{orig:<{col_width}} | {enh:<{col_width}}")
    
    result.append("=" * (col_width * 2 + 3))
    
    return '\n'.join(result)


def highlight_code(code: str, language: str = "") -> str:
    """
    Apply syntax highlighting to code if pygments is available.
    
    Args:
        code: Code to highlight
        language: Language for highlighting
        
    Returns:
        Highlighted code string
    """
    if not HAS_PYGMENTS:
        return code
    
    try:
        if not language:
            lexer = guess_lexer(code)
        else:
            lexer = get_lexer_by_name(language, stripall=True)
        
        formatter = TerminalFormatter()
        return pygments.highlight(code, lexer, formatter)
    except Exception:
        return code


def main():
    """Main function to run the prompt enhancer."""
    args = parse_args()
    
    # Get prompt from argument or file
    prompt = args.prompt
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
    
    # Create enhancer
    enhancer = PromptEnhancer(
        aggressive=args.aggressive,
        fix_code=args.fix_code,
        add_missing_words=args.add_missing_words
    )
    
    # Enhance prompt
    result = enhancer.enhance_prompt(prompt)
    
    # Format output
    if args.format == "json":
        output = json.dumps(result, indent=2)
    elif args.format == "diff":
        output = display_diff(result["original"], result["enhanced"])
    elif args.format == "side-by-side":
        output = display_side_by_side(result["original"], result["enhanced"], args.width)
    else:
        output = result["enhanced"]
    
    # Print summary (unless quiet mode)
    if not args.quiet:
        print(f"Enhancement complete with {result['changes']} changes:")
        for category, count in result["details"].items():
            if count > 0:
                print(f"  - {category.replace('_', ' ').title()}: {count}")
        print()
    
    # Output result
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result["enhanced"])
            if not args.quiet:
                print(f"Enhanced prompt saved to: {args.output}")
        except Exception as e:
            logger.error(f"Error writing output file: {e}")
    else:
        print(output)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhance text and code prompts")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--prompt", type=str, help="Text prompt to enhance")
    input_group.add_argument("--file", type=str, help="Path to file containing prompt")
    
    # Enhancement options
    parser.add_argument("--aggressive", action="store_true", 
                        help="Use more aggressive enhancement")
    parser.add_argument("--no-code-fix", dest="fix_code", action="store_false",
                        help="Don't fix code blocks")
    parser.add_argument("--no-add-words", dest="add_missing_words", action="store_false",
                        help="Don't add missing words")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--format", choices=["text", "json", "diff", "side-by-side"],
                        default="side-by-side", help="Output format")
    parser.add_argument("--width", type=int, default=80,
                        help="Width for side-by-side display")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress summary output")
    
    return parser.parse_args()


if __name__ == "__main__":
    main() 