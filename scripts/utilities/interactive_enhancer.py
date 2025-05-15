#!/usr/bin/env python3
"""
Interactive Prompt Enhancer

This script provides an interactive interface to the prompt_enhancer.
It takes user input, enhances it, and displays the enhanced version with
improvements for context awareness, spelling, syntax, and formatting.

Usage:
    python -m scripts.utilities.interactive_enhancer
    
    OR run from project root:
    python scripts/utilities/interactive_enhancer.py
"""

import sys
import os
import traceback
import re
import nltk
from textblob import TextBlob

# Add parent directory to path to allow imports when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    # First try relative import (when run as a module)
    from . import prompt_enhancer
    from .prompt_enhancer import PromptEnhancer, display_side_by_side
except ImportError:
    # Fall back to direct import (when run as a script)
    import scripts.utilities.prompt_enhancer as prompt_enhancer
    from scripts.utilities.prompt_enhancer import PromptEnhancer, display_side_by_side

# Import advanced NLP libraries for grammar checking
try:
    import language_tool_python
    HAS_LANGUAGE_TOOL = True
except ImportError:
    print("Note: language-tool-python not found. Install with: pip install language-tool-python")
    HAS_LANGUAGE_TOOL = False

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")  # Load large model for better accuracy
        HAS_SPACY_LG = True
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")  # Fall back to small model
            print("Note: Using smaller spaCy model. For better results: python -m spacy download en_core_web_lg")
            HAS_SPACY_LG = False
        except OSError:
            print("Note: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            nlp = None
    HAS_SPACY = nlp is not None
except ImportError:
    print("Note: spaCy not found. Install with: pip install spacy")
    HAS_SPACY = False
    HAS_SPACY_LG = False
    nlp = None

try:
    from gingerit.gingerit import GingerIt
    parser = GingerIt()
    HAS_GINGERIT = True
except ImportError:
    print("Note: gingerit not found. Install with: pip install gingerit")
    HAS_GINGERIT = False

# Ensure we have necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Fallback patterns for specific cases that might be missed
CRITICAL_GRAMMAR_PATTERNS = {
    # Common technical writing errors
    r'\bthe the\b': 'the',
    r'\bin the in the\b': 'in the',
    r'\ba a\b': 'a',
    r'\bis is\b': 'is',
    r'\bif if\b': 'if',
    r'\bend-to-end\b': 'end to end',
    
    # Technical terms that may be incorrectly corrected
    r'\b(could|would|should|must|might) of\b': lambda m: f"{m.group(1)} have",
    r'\bsuppose to\b': 'supposed to',
    r'\buse to\b': 'used to',
}

def fix_grammar(text):
    """
    Fix grammar issues using a hybrid approach with multiple tools.
    
    Args:
        text: Text to fix
        
    Returns:
        Text with corrected grammar
    """
    if not text or not isinstance(text, str):
        return text
        
    # Skip code blocks for grammar checking
    code_blocks = {}
    code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    
    def replace_code_blocks(match):
        lang, code = match.groups()
        placeholder = f"CODE_BLOCK_{len(code_blocks)}"
        code_blocks[placeholder] = f"```{lang}\n{code}\n```"
        return placeholder
    
    # Replace code blocks with placeholders
    text_without_code = code_block_pattern.sub(replace_code_blocks, text)
    
    # Step 1: Use language-tool-python for comprehensive grammar checking
    if HAS_LANGUAGE_TOOL:
        try:
            tool = language_tool_python.LanguageTool('en-US')
            # Avoid correcting potential code fragments
            text_without_code = tool.correct(text_without_code)
        except Exception as e:
            print(f"LanguageTool error: {e}")
    
    # Step 2: Use gingerit for additional grammar and spelling corrections
    if HAS_GINGERIT:
        try:
            # Process text in reasonable chunks to avoid API limits
            MAX_CHUNK_SIZE = 600  # Characters per chunk
            corrected_chunks = []
            
            # Split text into paragraphs
            paragraphs = text_without_code.split('\n')
            for paragraph in paragraphs:
                if len(paragraph) <= MAX_CHUNK_SIZE:
                    try:
                        result = parser.parse(paragraph)
                        corrected_chunks.append(result['result'])
                    except Exception:
                        corrected_chunks.append(paragraph)
                else:
                    # Split long paragraphs by sentence
                    sentences = nltk.sent_tokenize(paragraph)
                    corrected_sentences = []
                    
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= MAX_CHUNK_SIZE:
                            current_chunk += sentence + " "
                        else:
                            # Process accumulated chunk
                            if current_chunk:
                                try:
                                    result = parser.parse(current_chunk.strip())
                                    corrected_sentences.append(result['result'])
                                except Exception:
                                    corrected_sentences.append(current_chunk.strip())
                            # Start new chunk
                            current_chunk = sentence + " "
                    
                    # Process last chunk
                    if current_chunk:
                        try:
                            result = parser.parse(current_chunk.strip())
                            corrected_sentences.append(result['result'])
                        except Exception:
                            corrected_sentences.append(current_chunk.strip())
                    
                    corrected_chunks.append(" ".join(corrected_sentences))
            
            text_without_code = "\n".join(corrected_chunks)
        except Exception as e:
            print(f"Gingerit error: {e}")
    
    # Step 3: Use spaCy for contextual grammar correction
    if HAS_SPACY:
        try:
            doc = nlp(text_without_code)
            corrected_tokens = []
            
            for i, token in enumerate(doc):
                # Look ahead/behind for context
                prev_token = doc[i-1] if i > 0 else None
                next_token = doc[i+1] if i < len(doc)-1 else None
                
                # Fix "their" vs "there" vs "they're" based on context
                if token.text.lower() == "their":
                    if token.head.pos_ in ["VERB", "AUX"] or (next_token and next_token.pos_ in ["VERB", "AUX"]):
                        corrected_tokens.append("there")
                    elif next_token and (next_token.text.lower() in ["is", "are", "be", "was", "were", "going", "coming"]):
                        corrected_tokens.append("they're")
                    else:
                        corrected_tokens.append(token.text)
                
                # Fix "your" vs "you're" based on context
                elif token.text.lower() == "your":
                    if token.head.pos_ in ["VERB", "AUX"] or (next_token and next_token.pos_ in ["VERB", "AUX"]):
                        corrected_tokens.append("you're")
                    elif next_token and (next_token.text.lower() in ["is", "are", "be", "was", "were", "going", "coming"]):
                        corrected_tokens.append("you're")
                    else:
                        corrected_tokens.append(token.text)
                
                # Fix "its" vs "it's" based on context
                elif token.text.lower() == "its":
                    if token.head.pos_ in ["VERB", "AUX"] or (next_token and next_token.pos_ in ["VERB", "AUX"]):
                        corrected_tokens.append("it's")
                    elif next_token and (next_token.text.lower() in ["is", "are", "be", "was", "were", "going", "coming"]):
                        corrected_tokens.append("it's")
                    else:
                        corrected_tokens.append(token.text)
                
                else:
                    corrected_tokens.append(token.text)
            
            # Reconstruct text with proper spacing
            text_without_code = ""
            for i, token in enumerate(corrected_tokens):
                # Add space before token if needed
                if i > 0 and not (token in ".,;:!?)]}'" or corrected_tokens[i-1] in "([{\"'"):
                    text_without_code += " "
                text_without_code += token
            
        except Exception as e:
            print(f"spaCy error: {e}")
    
    # Step 4: Apply critical pattern fixes as a final polish
    for pattern, replacement in CRITICAL_GRAMMAR_PATTERNS.items():
        if callable(replacement):
            text_without_code = re.sub(pattern, replacement, text_without_code, flags=re.IGNORECASE)
        else:
            text_without_code = re.sub(pattern, replacement, text_without_code, flags=re.IGNORECASE)
    
    # Restore code blocks
    for placeholder, code_block in code_blocks.items():
        text_without_code = text_without_code.replace(placeholder, code_block)
    
    return text_without_code

def enhance_spelling_only(text):
    """
    Enhance spelling more aggressively than the default enhancer.
    
    Args:
        text: Text to fix spelling
        
    Returns:
        Text with corrected spelling
    """
    if not text or not isinstance(text, str):
        return text
        
    # Skip code blocks for spelling checking
    code_blocks = {}
    code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
    
    def replace_code_blocks(match):
        lang, code = match.groups()
        placeholder = f"CODE_BLOCK_{len(code_blocks)}"
        code_blocks[placeholder] = f"```{lang}\n{code}\n```"
        return placeholder
    
    # Replace code blocks with placeholders
    text_without_code = code_block_pattern.sub(replace_code_blocks, text)
    
    # Split into words but preserve structure
    # This pattern captures words, whitespace, and punctuation separately
    tokens = re.findall(r'(\s+|[^\w\s]+|\w+)', text_without_code)
    corrected_tokens = []
    
    # Create context window for better correction
    context_size = 2  # Words before and after for context
    
    for i, token in enumerate(tokens):
        # Only try to correct words (skip whitespace and punctuation)
        if re.match(r'^\w+$', token):
            # Skip URLs, technical terms, proper nouns, etc.
            if (token.lower() == token and     # Not capitalized mid-sentence
                not re.match(r'^[A-Z]{2,}$', token) and  # Not an acronym
                len(token) > 2 and          # Skip very short words
                not re.match(r'^\d+$', token)):  # Skip numbers
                
                # Get context words for better correction
                start_idx = max(0, i - context_size)
                end_idx = min(len(tokens), i + context_size + 1)
                context_words = [t for t in tokens[start_idx:end_idx] if re.match(r'^\w+$', t)]
                
                # Apply TextBlob spelling correction with context if possible
                try:
                    # If we have enough context, use it for correction
                    if len(context_words) > 1:
                        context_text = ' '.join(context_words)
                        blob = TextBlob(context_text)
                        corrected_context = str(blob.correct())
                        corrected_words = corrected_context.split()
                        
                        # Find position of our word in corrected context
                        original_position = context_words.index(token)
                        if original_position < len(corrected_words):
                            corrected = corrected_words[original_position]
                        else:
                            corrected = str(TextBlob(token).correct())
                    else:
                        corrected = str(TextBlob(token).correct())
                    
                    corrected_tokens.append(corrected)
                except Exception:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        else:
            # Keep whitespace and punctuation as is
            corrected_tokens.append(token)
    
    text_without_code = ''.join(corrected_tokens)
    
    # Restore code blocks
    for placeholder, code_block in code_blocks.items():
        text_without_code = text_without_code.replace(placeholder, code_block)
    
    return text_without_code

def detect_programming_language(code_text):
    """
    Try to detect programming language from code text to provide better syntax enhancements.
    
    Args:
        code_text: Text that might contain code
        
    Returns:
        Detected language name or empty string
    """
    # Simple language detection based on keywords and syntax patterns
    languages = {
        'python': [
            r'\bdef\s+\w+\s*\(', r'\bclass\s+\w+\s*:', r'\bimport\s+\w+', 
            r'\bif\s+.*:', r'\bfor\s+.*:', r'\bwhile\s+.*:', r'\bprint\s*\('
        ],
        'javascript': [
            r'\bfunction\s+\w+\s*\(', r'\bconst\s+\w+\s*=', r'\blet\s+\w+\s*=',
            r'\bvar\s+\w+\s*=', r'\bconsole\.log\s*\(', r'\b\w+\s*=>\s*{'
        ],
        'java': [
            r'\bpublic\s+class\s+\w+', r'\bprivate\s+\w+\s+\w+', r'\bSystem\.out\.println',
            r'\bpublic\s+static\s+void\s+main'
        ],
        'html': [
            r'<html', r'<div', r'<body', r'<script', r'<head', r'</\w+>'
        ]
    }
    
    # Count matches for each language
    scores = {lang: 0 for lang in languages}
    
    for lang, patterns in languages.items():
        for pattern in patterns:
            matches = re.findall(pattern, code_text)
            scores[lang] += len(matches)
    
    # Return the language with the highest score if it exceeds a threshold
    best_lang = max(scores, key=scores.get)
    if scores[best_lang] >= 2:  # Threshold to confirm it's actually code
        return best_lang
    return ""

def main():
    """Main function for interactive prompt enhancer."""
    print("Welcome to Interactive Prompt Enhancer!")
    print("Enter your text to enhance. Type 'exit' or 'quit' to exit.")
    print("Type 'help' for options. Press Ctrl+D (EOF) to submit multiline input.\n")
    
    # Create enhancer with default settings
    enhancer = PromptEnhancer(
        aggressive=False,
        fix_code=True,
        add_missing_words=True
    )
    
    # Add additional settings
    spell_check_mode = "normal"  # Can be "normal", "aggressive", or "off"
    context_aware = True  # Context-aware enhancements
    grammar_check = True  # Grammar checking
    
    # Display NLP engine availability
    print("\nAvailable NLP engines:")
    print(f"  Language Tool: {'✓' if HAS_LANGUAGE_TOOL else '✗'}")
    print(f"  spaCy: {'✓ (large model)' if HAS_SPACY_LG else '✓ (small model)' if HAS_SPACY else '✗'}")
    print(f"  Gingerit: {'✓' if HAS_GINGERIT else '✗'}")
    print(f"  TextBlob: ✓")
    
    # Main interaction loop
    while True:
        print("\n" + "-" * 50)
        print("Enter text (Ctrl+D to submit):")
        
        # Collect multiline input with better error handling
        lines = []
        try:
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    # End of input
                    break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\nError reading input: {e}")
            continue
        
        # Join lines to form the prompt
        prompt = "\n".join(lines)
        
        # Check for exit commands
        if prompt.strip().lower() in ('exit', 'quit'):
            print("Exiting...")
            break
        
        # Check for help command
        if prompt.strip().lower() == 'help':
            print("\nCommands:")
            print("  exit, quit - Exit the program")
            print("  help - Show this help message")
            print("  aggressive - Toggle aggressive enhancement")
            print("  no-code - Toggle code fixing")
            print("  no-words - Toggle adding missing words")
            print("  context - Toggle context-aware enhancement")
            print("  grammar - Toggle grammar checking")
            print("  spell-mode - Toggle between spell check modes (normal/aggressive/off)")
            print("  settings - Show current settings")
            continue
        
        # Check for settings commands
        if prompt.strip().lower() == 'aggressive':
            enhancer.aggressive = not enhancer.aggressive
            print(f"Aggressive enhancement: {'ON' if enhancer.aggressive else 'OFF'}")
            continue
            
        if prompt.strip().lower() == 'no-code':
            enhancer.fix_code = not enhancer.fix_code
            print(f"Code fixing: {'ON' if enhancer.fix_code else 'OFF'}")
            continue
            
        if prompt.strip().lower() == 'no-words':
            enhancer.add_missing_words = not enhancer.add_missing_words
            print(f"Adding missing words: {'ON' if enhancer.add_missing_words else 'OFF'}")
            continue
        
        if prompt.strip().lower() == 'context':
            context_aware = not context_aware
            print(f"Context-aware enhancement: {'ON' if context_aware else 'OFF'}")
            continue
            
        if prompt.strip().lower() == 'grammar':
            grammar_check = not grammar_check
            print(f"Grammar checking: {'ON' if grammar_check else 'OFF'}")
            continue
            
        if prompt.strip().lower() == 'spell-mode':
            # Cycle through spell check modes
            if spell_check_mode == "normal":
                spell_check_mode = "aggressive"
            elif spell_check_mode == "aggressive":
                spell_check_mode = "off"
            else:
                spell_check_mode = "normal"
            print(f"Spell check mode: {spell_check_mode.upper()}")
            continue
            
        if prompt.strip().lower() == 'settings':
            print("\nCurrent settings:")
            print(f"  Aggressive enhancement: {'ON' if enhancer.aggressive else 'OFF'}")
            print(f"  Code fixing: {'ON' if enhancer.fix_code else 'OFF'}")
            print(f"  Adding missing words: {'ON' if enhancer.add_missing_words else 'OFF'}")
            print(f"  Context-aware enhancement: {'ON' if context_aware else 'OFF'}")
            print(f"  Grammar checking: {'ON' if grammar_check else 'OFF'}")
            print(f"  Spell check mode: {spell_check_mode.upper()}")
            continue
        
        # If input is empty, continue
        if not prompt.strip():
            continue
        
        try:
            # Track grammar changes for reporting
            grammar_changes = 0
            original_prompt = prompt
            
            # Detect if input contains code and which language
            code_language = ""
            if enhancer.fix_code and "```" in prompt:
                # Extract code blocks
                code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', prompt, re.DOTALL)
                if code_blocks:
                    for lang, code in code_blocks:
                        if not lang:  # If language not specified in markdown
                            detected_lang = detect_programming_language(code)
                            if detected_lang:
                                # Replace the code block with language specified
                                prompt = prompt.replace(f"```\n{code}\n```", 
                                                      f"```{detected_lang}\n{code}\n```")
                                print(f"Detected code language: {detected_lang}")
            
            # Apply grammar fixing if enabled
            if grammar_check:
                before_grammar = prompt
                prompt = fix_grammar(prompt)
                # Count approximate number of changes
                if before_grammar != prompt:
                    # Simple diff-based change count
                    grammar_changes = sum(1 for a, b in zip(before_grammar.split(), prompt.split()) if a != b)
            
            # Apply enhancement based on spell check mode and context awareness
            if spell_check_mode == "aggressive":
                # First apply aggressive spelling correction
                corrected_prompt = enhance_spelling_only(prompt) if context_aware else prompt
                # Then pass to regular enhancer
                result = enhancer.enhance_prompt(corrected_prompt)
                # Add original for comparison
                result["original"] = original_prompt
            elif spell_check_mode == "off":
                # Create a custom result without spell checking
                enhanced_text = prompt
                # Still apply other enhancements if needed
                result = enhancer.enhance_prompt(prompt)
                result["details"]["spelling"] = 0  # Zero out spelling changes
            else:
                # Normal mode - use the enhancer directly
                result = enhancer.enhance_prompt(prompt)
                
            # Add grammar changes to the result
            if grammar_check and grammar_changes > 0:
                if "grammar" not in result["details"]:
                    result["details"]["grammar"] = grammar_changes
                else:
                    result["details"]["grammar"] += grammar_changes
                
                # Update total changes count
                result["changes"] += grammar_changes
            
            # Print summary
            print(f"\nEnhancement complete with {result['changes']} changes:")
            for category, count in result["details"].items():
                if count > 0:
                    print(f"  - {category.replace('_', ' ').title()}: {count}")
            
            # Display side by side comparison
            print("\n" + "=" * 80)
            print(display_side_by_side(result["original"], result["enhanced"], width=80))
            print("=" * 80)
        
        except Exception as e:
            print(f"\nError enhancing prompt: {str(e)}")
            if enhancer.aggressive:
                print(traceback.format_exc())

if __name__ == "__main__":
    main()