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


def remove_length_diacritics(ipa_string):
    """Remove length diacritics from the IPA string"""
    return "".join(c for c in ipa_string if c not in {"ː", "ˑ"})


def filter_chars(ipa_string, filter_type="cns_vwl_str_len_wb_sb", remap_rhotic=True):
    """Filter characters to only include any of FILTERS.

    Optional:
    - remove_tie: if "_rmv_tie" in filter_type
    - remap_rhotic: replace ɚ and ɝ with əɹ
    """
    remove_tie = filter_type.endswith("_rmv_tie")
    if remove_tie:
        filter_type = filter_type[: -len("_rmv_tie")]
    if "˧" in ipa_string:
        raise ValueError(
            "Warning: we use this IPA ˧ as a temporary marker for ŋ̍ which is an unsupported IPA symbol,any ˧ will get mapped to ŋ̍."
        )
    # temporarily replace with a tone marker as a placeholder
    if "ŋ̍" in ipa_string:
        ipa_string = ipa_string.replace("ŋ̍", "˧")

    ipa = str(
        IPAString(unicode_string=ipa_string)
        .filter_chars(filter_type)
        .canonical_representation
    )
    # map back to the original syllabic n-g
    if "˧" in ipa:
        ipa = ipa.replace("˧", "ŋ̍")
    if remove_tie:
        ipa = "".join({"͡": ""}.get(c, c) for c in ipa)

    if remap_rhotic:
        ipa = ipa.replace("ɚ", "əɹ").replace("ɝ", "əɹ")

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
