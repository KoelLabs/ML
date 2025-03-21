#!/usr/bin/env python3

# utils for processing text

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.codes import arpabet2ipa

from string import punctuation
import g2p_en

g2p = g2p_en.G2p()

def english2ipa(text):
    arpa = g2p(text)
    def arpa2ipa(a):
        if a == ' ':
            return ' '
        try:
            return arpabet2ipa(a)
        except:
            return a
    return ''.join(map(arpa2ipa, arpa))

def remove_punctuation(text):
    return ''.join([c for c in text if c not in punctuation])

def main(args):
    if len(args) < 2:
        print("Usage: python ./scripts/core/text.py <command> <text>")
        return
    command = args[0]
    text = args[1]
    if command == "english2ipa":
        print(english2ipa(text))
    elif command == "remove_punctuation":
        print(remove_punctuation(text))
    else:
        print("Unknown command")

if __name__ == "__main__":
    main(sys.argv[1:])
