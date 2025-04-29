#!/usr/bin/env python3

# IPA string manipulation functions
# IPA copy paste: https://westonruter.github.io/ipa-chart/keyboard/

import sys

import ipapy
from ipapy.ipastring import IPAString

FILTERS = set(
    (
        "rmv_tie",  # remove tie bar (append to any other filter)
        "consonants",  # consonants
        "vowels",  # vowels
        "letters",  # consonants and vowels
        "cns_vwl_pstr",  # consonants, vowels, primary stress diacritics
        "cns_vwl_pstr_long",  # consonants, vowels, primary stress diacritics, long suprasegmentals
        "cns_vwl_str",  # consonants, vowels, stress diacritics
        "cns_vwl_str_len",  # consonants, vowels, stress diacritics, length diacritics
        "cns_vwl_str_len_wb",  # consonants, vowels, stress diacritics, length diacritics, word breaks
        "cns_vwl_str_len_wb_sb",  # consonants, vowels, stress diacritics, length diacritics, word breaks, syllable breaks
    )
)


def canonize(ipa_string, ignore=False):
    """canonize the Unicode representation of the IPA string"""
    return str(
        IPAString(unicode_string=ipa_string, ignore=ignore).canonical_representation
    )


def remove_length_diacritics(ipa_string: str):
    """Remove length diacritics from the IPA string"""
    return "".join(c for c in ipa_string if c not in {"ː", "ˑ"})


def remove_tones_and_stress(ipa_string: str):
    """Remove tones and stress from the IPA string"""
    tones_and_stresss = ["˨˩h", "˧ʔ˥", "˧˨ʔ", "ˈ", "ˌ", "˥", "˧", "˨", "˩", "˦"]  # fmt: skip
    for tone in tones_and_stresss:
        ipa_string = ipa_string.replace(tone, "")
    return ipa_string


def remove_tie_marker(ipa_string: str):
    """Remove tie marker from the IPA string"""
    return "".join({"͡": ""}.get(c, c) for c in ipa_string)


def simplify_ipa(ipa_string: str):
    """Simplify the IPA string by removing length markers, ties, expanding rhotics, and using the most common symbol for each sound"""

    if "̄" in ipa_string:
        raise ValueError("Warning: we use this IPA as a temporary marker for ŋ̍")
    if "ŋ̍" in ipa_string:
        ipa_string = ipa_string.replace("ŋ̍", "̄")

    ipa = str(
        IPAString(unicode_string=ipa_string, ignore=True).canonical_representation
    )

    ipa = ipa.replace("̄", "ŋ̍")
    ipa = ipa.replace(" ", "")
    ipa = remove_tie_marker(ipa)
    ipa = remove_length_diacritics(ipa)
    ipa = remove_tones_and_stress(ipa)
    ipa = ipa.replace("ɚ", "əɹ").replace("ɝ", "əɹ")

    return ipa


def filter_chars(ipa_string, filter_type="cns_vwl_str_len_wb_sb"):
    """Filter characters to only include any of FILTERS."""
    remove_tie = filter_type.endswith("_rmv_tie")
    if remove_tie:
        filter_type = filter_type[: -len("_rmv_tie")]

    ipa = str(
        IPAString(unicode_string=ipa_string)
        .filter_chars(filter_type)
        .canonical_representation
    )

    if remove_tie:
        ipa = remove_tie_marker(ipa)

    return ipa


def is_equivalent(ipa_string1, ipa_string2):
    """Check if two IPA strings are equivalent"""
    return IPAString(unicode_string=ipa_string1).is_equivalent(ipa_string2)


def is_valid_ipa(ipa_string):
    """Check if the given Unicode string is a valid IPA string"""
    return ipapy.is_valid_ipa(ipa_string)


def clean(ipa_string):
    """Remove characters that are not IPA valid"""
    return "".join(ipapy.remove_invalid_ipa_characters(ipa_string))  # type: ignore


def _describe_symbol(symbol):
    description = (
        symbol.canonical_representation.replace("suprasegmental", "")
        .replace("diacritic", "")
        .replace("word-break", "")
    )
    if description.strip() == "":
        return "silence"
    else:
        return description


def describe(ipa_string):
    """Describe the IPA string"""
    return [
        (symbol, _describe_symbol(symbol))
        for symbol in IPAString(unicode_string=ipa_string)
    ]


def usage():
    print("Usage: python ./scripts/core/ipa.py check <ipa_string> ...")
    print("Usage: python ./scripts/core/ipa.py clean <ipa_string> ...")
    print("Usage: python ./scripts/core/ipa.py canonize <ipa_string> ...")
    print("Usage: python ./scripts/core/ipa.py simplify <ipa_string> ...")
    print("Usage: python ./scripts/core/ipa.py filter [<filter_type>] <ipa_string> ...")
    print("Usage: python ./scripts/core/ipa.py cmp <ipa_string1> <ipa_string2>")
    print("Usage: python ./scripts/core/ipa.py describe <ipa_string>")
    print("Supported filters:", FILTERS)


def main(args):
    action = args.pop(0)

    if action == "check":
        for arg in args:
            print(arg, "=>", is_valid_ipa(arg))
    elif action == "clean":
        for arg in args:
            print(arg, "=>", clean(arg))
    elif action == "canonize":
        for arg in args:
            print(arg, "=>", canonize(arg))
    elif action == "simplify":
        for arg in args:
            print(arg, "=>", simplify_ipa(arg))
    elif action == "filter":
        if len(args) < 2:
            filter_type = "cns_vwl_str_len_wb_sb"
        else:
            filter_type = args.pop(0)
        for arg in args:
            print(arg, "=>", filter_chars(arg, filter_type))
    elif action == "cmp":
        print(args[0], args[1], "=>", is_equivalent(args[0], args[1]))
    elif action == "describe":
        arg = " ".join(args)
        for sym, desc in describe(arg):
            print(sym, desc)
    else:
        print("Invalid action")
        usage()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(e)
        usage()
