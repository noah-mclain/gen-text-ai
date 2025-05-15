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

# Ensure we have necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Define common grammatical error patterns
HOMOPHONE_CORRECTIONS = {
    # their/there/they're
    r'\b(their)\b(?=\s+(?:is|are|was|were|has|have|will|would|should|could|to be))\b': 'there',
    r'\b(their)\b(?=\s+(?:going|coming|leaving|walking|running))\b': 'they\'re',
    r'\b(there)\b(?=\s+(?:book|house|car|dog|cat|children|parents|friend|computer|money|things|stuff|clothes|shoes|toys|games)s?\b)': 'their',
    r'\b(there)\b(?=\s+(?:going|walking|running|leaving))\b': 'they\'re',
    
    # your/you're
    r'\b(your)\b(?=\s+(?:welcome|going|coming|looking|the best|awesome|amazing|great|wonderful|fantastic|beautiful|smart|intelligent|kind|nice|sweet|cute|adorable))\b': 'you\'re',
    r'\b(you\'re)\b(?=\s+(?:house|car|dog|cat|children|parents|friend|computer|money|things|stuff|clothes|shoes|toys|games)s?\b)': 'your',
    
    # its/it's
    r'\b(its)\b(?=\s+(?:going|coming|getting|a|an|the|not|very|really|so|too|quite))\b': 'it\'s',
    r'\b(it\'s)\b(?=\s+(?:owner|color|size|shape|form|function|purpose|name|title|content|meaning))\b': 'its',
    
    # to/too/two
    r'\b(to)\b(?=\s+(?:much|many|little|few|late|soon|bad|good|hot|cold))\b': 'too',
    r'\b(too)\b(?=\s+(?:go|come|get|see|hear|feel|taste|smell|do|make|finish|start|begin|end))\b': 'to',
    
    # than/then
    r'\b(than)\b(?=\s+(?:he|she|it|they|we|I|you|the|a|an|this|that|these|those|my|your|his|her|its|our|their)\s+(?:went|came|got|said|told|asked|answered|replied|shouted|whispered|called|responded))\b': 'then',
    r'\b(then)\b(?=\s+(?:better|worse|more|less|bigger|smaller|larger|higher|lower|stronger|weaker|faster|slower|easier|harder|simpler|more complex|more difficult|more challenging))\b': 'than',
    
    # affect/effect
    r'\b(effect)\b(?=\s+(?:the|a|an|my|your|his|her|its|our|their)\s+(?:decision|performance|ability|capacity|potential|outcome|result))\b': 'affect',
    r'\b(affect)\b(?=\s+(?:of|on|upon))\b': 'effect',
    
    # accept/except
    r'\b(accept)\b(?=\s+(?:for|from|when|if))\b': 'except',
    r'\b(except)\b(?=\s+(?:the|a|an|my|your|his|her|its|our|their|this|that|these|those)\s+(?:offer|invitation|proposal|terms|conditions|contract|deal|agreement))\b': 'accept',
    
    # advice/advise
    r'\b(advice)\b(?=\s+(?:him|her|them|me|you|us|the|a|an|my|your|his|her|its|our|their|this|that|these|those)\s+(?:to|not to|about|on|regarding|concerning))\b': 'advise',
    r'\b(advise)\b(?=\s+(?:is|was|has been|will be|would be|could be|should be|might be))\b': 'advice',
    
    # weather/whether
    r'\b(weather)\b(?=\s+(?:or not|to|we should|they should|he should|she should|I should|you should))\b': 'whether',
    r'\b(whether)\b(?=\s+(?:is|was|has been|will be|would be|could be|should be|might be)\s+(?:nice|good|bad|terrible|horrible|awful|wonderful|beautiful|perfect|great|amazing|fantastic|cloudy|sunny|rainy|snowy|windy|stormy))\b': 'weather',
    
    # complement/compliment
    r'\b(complement)\b(?=\s+(?:him|her|them|me|you|us|the|a|an|my|your|his|her|its|our|their|this|that|these|those)\s+(?:on|about|for))\b': 'compliment',
    r'\b(compliment)\b(?=\s+(?:to|the|a|an|this|that|these|those)\s+(?:color|flavors|tastes|sounds|colors|styles|designs|patterns|textures))\b': 'complement',
}

# Grammar patterns for other common errors
GRAMMAR_CORRECTIONS = {
    # subject-verb agreement
    r'\b(I|you|we|they)\s+(is|was)\b': lambda m: f"{m.group(1)} {'am' if m.group(1) == 'I' else 'are' if m.group(1) != 'you' else 'were' if m.group(2) == 'was' else 'are'}",
    r'\b(he|she|it)\s+(am|are|were)\b': lambda m: f"{m.group(1)} {'is' if m.group(2) in ('am', 'are') else 'was'}",
    
    # double negatives
    r'\b(don\'t|doesn\'t|didn\'t|can\'t|couldn\'t|won\'t|wouldn\'t|shouldn\'t|haven\'t|hasn\'t|hadn\'t)\s+.{1,20}\s+(no|none|nobody|nothing|nowhere|never)\b': lambda m: m.group(0).replace(m.group(1), m.group(1).replace("n't", "")),
    
    # a vs an
    r'\b(a)\s+([aeiouAEIOU][a-zA-Z]+)\b': lambda m: f"an {m.group(2)}",
    r'\b(an)\s+([^aeiouAEIOU][a-zA-Z]+)\b': lambda m: f"a {m.group(2)}",
    
    # common phrases
    r'\b(could|would|should|must|might) of\b': lambda m: f"{m.group(1)} have",
    r'\bin regards to\b': 'regarding',
    r'\bfor all intensive purposes\b': 'for all intents and purposes',
    r'\bsuppose to\b': 'supposed to',
    r'\buse to\b': 'used to',
    r'\ba lot\b': 'a lot',
}

def fix_grammar(text):
    """
    Fix common grammatical errors in text.
    
    Args:
        text: Text to fix
        
    Returns:
        Text with corrected grammar
    """
    if not text or not isinstance(text, str):
        return text
    
    # Tag parts of speech for better grammar checking
    try:
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        pos_tags = []
    
    # Fix homophone errors
    for pattern, replacement in HOMOPHONE_CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Fix other grammar errors
    for pattern, replacement in GRAMMAR_CORRECTIONS.items():
        if callable(replacement):
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        else:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Check for possessive apostrophes
    text = re.sub(r'\b(\w+)s\b(?=\s+(?:house|car|dog|cat|book|phone|computer|office|desk|chair|table|bed|room|apartment|home|property|money|wallet|purse|bag|coat|jacket|shirt|pants|shoes|hat|glasses|watch|ring|necklace|earrings))', r"\1's", text)
    
    return text

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
        
    # Split into words but preserve structure
    # This pattern captures words, whitespace, and punctuation separately
    tokens = re.findall(r'(\s+|[^\w\s]+|\w+)', text)
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
    
    return ''.join(corrected_tokens)

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
                prompt = fix_grammar(prompt)
            
            # Apply enhancement based on spell check mode and context awareness
            if spell_check_mode == "aggressive":
                # First apply aggressive spelling correction
                corrected_prompt = enhance_spelling_only(prompt) if context_aware else prompt
                # Then pass to regular enhancer
                result = enhancer.enhance_prompt(corrected_prompt)
                # Add original for comparison
                result["original"] = prompt
            elif spell_check_mode == "off":
                # Create a custom result without spell checking
                enhanced_text = prompt
                # Still apply other enhancements if needed
                result = enhancer.enhance_prompt(prompt)
                result["details"]["spelling"] = 0  # Zero out spelling changes
            else:
                # Normal mode - use the enhancer directly
                result = enhancer.enhance_prompt(prompt)
                
                # Add grammar changes to the result if enabled
                if grammar_check and "grammar" not in result["details"]:
                    result["details"]["grammar"] = sum(1 for pattern in HOMOPHONE_CORRECTIONS 
                                                    if re.search(pattern, prompt, re.IGNORECASE))
                    result["details"]["grammar"] += sum(1 for pattern in GRAMMAR_CORRECTIONS 
                                                     if re.search(pattern, prompt, re.IGNORECASE))
            
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