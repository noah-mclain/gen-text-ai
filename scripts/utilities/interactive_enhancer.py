#!/usr/bin/env python3
"""
Interactive Prompt Enhancer

This script provides an interactive interface to the prompt_enhancer.
It takes user input, enhances it, and displays the enhanced version.

Usage:
    python -m scripts.utilities.interactive_enhancer
"""

import sys
import os
from scripts.utilities.prompt_enhancer import PromptEnhancer, display_side_by_side
from textblob import TextBlob
import re

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
    
    for token in tokens:
        # Only try to correct words (skip whitespace and punctuation)
        if re.match(r'^\w+$', token):
            # Skip URLs, technical terms, proper nouns, etc.
            if (token.lower() == token and     # Not capitalized mid-sentence
                not re.match(r'^[A-Z]{2,}$', token) and  # Not an acronym
                len(token) > 2 and          # Skip very short words
                not re.match(r'^\d+$', token)):  # Skip numbers
                
                # Apply TextBlob spelling correction
                try:
                    corrected = str(TextBlob(token).correct())
                    corrected_tokens.append(corrected)
                except:
                    corrected_tokens.append(token)
            else:
                corrected_tokens.append(token)
        else:
            # Keep whitespace and punctuation as is
            corrected_tokens.append(token)
    
    return ''.join(corrected_tokens)

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
    
    # Main interaction loop
    while True:
        print("\n" + "-" * 50)
        print("Enter text (Ctrl+D to submit):")
        
        # Collect multiline input
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            # End of input
            pass
        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\nExiting...")
            sys.exit(0)
        
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
            print(f"  Spell check mode: {spell_check_mode.upper()}")
            continue
        
        # If input is empty, continue
        if not prompt.strip():
            continue
        
        # Apply enhancement based on spell check mode
        if spell_check_mode == "aggressive":
            # First apply aggressive spelling correction
            corrected_prompt = enhance_spelling_only(prompt)
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
        
        # Print summary
        print(f"\nEnhancement complete with {result['changes']} changes:")
        for category, count in result["details"].items():
            if count > 0:
                print(f"  - {category.replace('_', ' ').title()}: {count}")
        
        # Display side by side comparison
        print("\n" + "=" * 80)
        print(display_side_by_side(result["original"], result["enhanced"], width=80))
        print("=" * 80)

if __name__ == "__main__":
    main() 