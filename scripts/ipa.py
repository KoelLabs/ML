import sys
import os

import ipapy
from ipapy.ipastring import IPAString
from ipapy.kirshenbaummapper import KirshenbaumMapper

kmapper = KirshenbaumMapper()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(os.path.dirname(__file__), "../repos/phonecodes/src"))
from repos.phonecodes.src import phonecodes

CODES = set(("ipa", "arpabet", "xsampa", "disc", "callhome"))
LANGUAGES = set(("eng", "deu", "nld", "arz", "cmn", "spa", "yue", "lao", "vie"))
FILTERS = set(
    (
        "consonants",
        "vowels",
        "letters",
        "cns_vwl_pstr",
        "cns_vwl_pstr_long",
        "cns_vwl_str",
        "cns_vwl_str_len",
        "cns_vwl_str_len_wb",
        "cns_vwl_str_len_wb_sb",
    )
)


#####################################################################
# X-SAMPA
def ipa2xsampa(ipa_string, language="eng"):
    return phonecodes.ipa2xsampa(ipa_string, language)


def xsampa2ipa(xsampa_string, language="eng"):
    return phonecodes.xsampa2ipa(xsampa_string, language)


#####################################################################
# DISC, the system used by CELEX
def ipa2disc(ipa_string, language="eng"):
    return phonecodes.ipa2disc(ipa_string, language)


def disc2ipa(disc_string, language="eng"):
    return phonecodes.disc2ipa(disc_string, language)


#####################################################################
# Kirshenbaum
def ipa2kirshenbaum(ipa_string, language="eng"):
    return kmapper.map_unicode_string(ipa_string)


def kirshenbaum2ipa(kirshenbaum_string, language="eng"):
    raise NotImplementedError


#######################################################################
# Callhome phone codes
def ipa2callhome(ipa_string, language="eng"):
    return phonecodes.ipa2callhome(ipa_string, language)


def callhome2ipa(callhome_string, language="eng"):
    return phonecodes.callhome2ipa(callhome_string, language)


#########################################################################
# ARPABET
def ipa2arpabet(ipa_string, language="eng"):
    return phonecodes.ipa2arpabet(ipa_string, language)


def arpabet2ipa(arpabet_string, language="eng"):
    return phonecodes.arpabet2ipa(arpabet_string, language)


#########################################################################
# TIMIT
def ipa2timit(ipa_string, language="eng"):
    raise NotImplementedError


def timit2ipa(timit_string, language="eng"):
    return phonecodes.timit2ipa(timit_string, language)


#########################################################################
# IPA utilities


def canonize(ipa_string):
    """canonize the Unicode representation of the IPA string"""
    return IPAString(unicode_string=ipa_string).canonical_representation


def filter_chars(ipa_string, filter_type="cns_vwl_str_len_wb_sb"):
    """Filter characters to only include consonants, vowels, letters, cns_vwl_pstr, cns_vwl_pstr_long, cns_vwl_str, cns_vwl_str_len, cns_vwl_str_len_wb, cns_vwl_str_len_wb_sb"""
    return str(
        IPAString(unicode_string=ipa_string)
        .filter_chars(filter_type)
        .canonical_representation
    )


def is_equivalent(ipa_string1, ipa_string2):
    """Check if two IPA strings are equivalent"""
    return IPAString(unicode_string=ipa_string1).is_equivalent(ipa_string2)


def is_valid_ipa(ipa_string):
    """Check if the given Unicode string is a valid IPA string"""
    return ipapy.is_valid_ipa(ipa_string)


def clean(ipa_string):
    """Remove characters that are not IPA valid"""
    return "".join(ipapy.remove_invalid_ipa_characters(ipa_string))


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
    print("Usage: python ./scripts/ipa.py check <ipa_string> ...")
    print("Usage: python ./scripts/ipa.py clean <ipa_string> ...")
    print("Usage: python ./scripts/ipa.py canonize <ipa_string> ...")
    print("Usage: python ./scripts/ipa.py filter [<filter_type>] <ipa_string> ...")
    print("Usage: python ./scripts/ipa.py cmp <ipa_string1> <ipa_string2>")
    print("Usage: python ./scripts/ipa.py convert <src> <tgt> <ipa_string>")
    print("Usage: python ./scripts/ipa.py describe <ipa_string>")
    print("Supported filters:", FILTERS)
    print("Supported codes:", CODES)
    print("Supported languages:", LANGUAGES)


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
    elif action == "convert":
        src, tgt, ipa_string = args
        if src == "ipa":
            print(ipa_string, "=>", globals()[f"ipa2{tgt}"](ipa_string))
        elif tgt == "ipa":
            print(ipa_string, "=>", globals()[f"{src}2ipa"](ipa_string))
        else:
            print(
                ipa_string,
                "=>",
                globals()[f"{src}2ipa"](globals()[f"ipa2{src}"](ipa_string)),
            )
    else:
        print("Invalid action")
        usage()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        raise e
        print(e)
        usage()
