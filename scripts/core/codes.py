#!/usr/bin/env python3

# Conversion between different phonetic codes
# Modified from https://github.com/jhasegaw/phonecodes/blob/master/src/phonecodes.py

import sys

# CODES = set(("ipa", "timit", "arpabet", "xsampa", "disc", "callhome"))
CODES = set(("ipa", "timit", "arpabet"))


def convert(phoneme_string, from_code, to_code):
    assert from_code in CODES, f"from_code must be one of {CODES}"
    assert to_code in CODES, f"to_code must be one of {CODES}"

    if from_code == "ipa":
        return globals()[f"ipa2{to_code}"](phoneme_string)
    elif to_code == "ipa":
        return globals()[f"{from_code}2ipa"](phoneme_string)
    else:
        return globals()[f"ipa2{to_code}"](
            globals()[f"{from_code}2ipa"](phoneme_string)
        )


def string2symbols(string, symbols):
    """Converts a string of symbols into a list of symbols, minimizing the number of untranslatable symbols, then minimizing the number of translated symbols."""
    N = len(string)
    symcost = 1  # path cost per translated symbol
    oovcost = len(string)  # path cost per untranslatable symbol
    maxsym = max(len(k) for k in symbols)  # max input symbol length
    # (pathcost to s[(n-m):n], n-m, translation[s[(n-m):m]], True/False)
    lattice = [(0, 0, "", True)]
    for n in range(1, N + 1):
        # Initialize on the assumption that s[n-1] is untranslatable
        lattice.append((oovcost + lattice[n - 1][0], n - 1, string[(n - 1) : n], False))
        # Search for translatable sequences s[(n-m):n], and keep the best
        for m in range(1, min(n + 1, maxsym + 1)):
            if (
                string[(n - m) : n] in symbols
                and symcost + lattice[n - m][0] < lattice[n][0]
            ):
                lattice[n] = (
                    symcost + lattice[n - m][0],
                    n - m,
                    string[(n - m) : n],
                    True,
                )
    # Back-trace
    tl = []
    translated = []
    n = N
    while n > 0:
        tl.append(lattice[n][2])
        translated.append(lattice[n][3])
        n = lattice[n][1]
    return (tl[::-1], translated[::-1])


#####################################################################
# Handle tones/stress markers
# fmt: off
TONE2IPA = {                                                                          
    'arz' : { '0':'', '1':'ˈ', '2':'ˌ',  '3': '',   '4': '',   '5': '',   '6': '' },  
    'eng' : { '0':'', '1':'ˈ', '2':'ˌ',  '3': '',   '4': '',   '5': '',   '6': '' },  
    'yue' : { '0':'', '1':'˥', '2':'˧˥', '3':'˧',   '4':'˨˩',  '5':'˩˧',  '6':'˨' },  
    'lao' : { '0':'', '1':'˧', '2':'˥˧', '3':'˧˩',  '4':'˥',   '5':'˩˧',  '6':'˩' },  
    'cmn' : { '0':'', '1':'˥', '2':'˧˥', '3':'˨˩˦', '4':'˥˩',  '5': '',   '6': '' },  
    'spa' : { '0':'', '1':'ˈ', '2':'ˌ',  '3': '',   '4': '',   '5': '',   '6': '' },  
    'vie' : { '0':'', '1':'˧', '2':'˨˩h', '3':'˧˥', '4':'˨˩˨', '5':'˧ʔ˥', '6':'˧˨ʔ' },
}                                                                                                    
IPA2TONE = {key: {v: k for k, v in val.items()} for key, val in TONE2IPA.items()}   
# fmt: on


def update_dict_with_tones(code2ipa: dict, ipa2code: dict, lang):
    code2ipa.update(TONE2IPA[lang])
    ipa2code.update(IPA2TONE[lang])


#####################################################################
# X-SAMPA
def ipa2xsampa(ipa_string, lang="eng"):
    raise NotImplementedError


def xsampa2ipa(xsampa_string, lang="eng"):
    raise NotImplementedError


#####################################################################
# DISC, the system used by CELEX
def ipa2disc(ipa_string, lang="eng"):
    raise NotImplementedError


def disc2ipa(disc_string, lang="eng"):
    raise NotImplementedError


#####################################################################
# Kirshenbaum
def ipa2kirshenbaum(ipa_string, lang="eng"):
    raise NotImplementedError


def kirshenbaum2ipa(kirshenbaum_string, lang="eng"):
    raise NotImplementedError


#######################################################################
# Callhome phone codes
def ipa2callhome(ipa_string, lang="eng"):
    raise NotImplementedError


def callhome2ipa(callhome_string, lang="eng"):
    raise NotImplementedError


#########################################################################
# Buckeye
BUCKEYE2IPA = {'aa':'ɑ', 'ae':'æ', 'ay':'aɪ', 'aw':'aʊ', 'ao':'ɔ', 'oy':'ɔɪ', 'ow':'oʊ', 'eh':'ɛ', 'ey':'eɪ', 'er':'ɝ', 'ah':'ʌ', 'uw':'u', 'uh':'ʊ', 'ih':'ɪ', 'iy':'i', 'm':'m', 'n':'n', 'en':'n̩', 'ng':'ŋ', 'l':'l', 'el':'l̩', 't':'t', 'd':'d', 'ch':'tʃ', 'jh':'dʒ', 'th':'θ', 'dh':'ð', 'sh':'ʃ', 'zh':'ʒ', 's':'s', 'z':'z', 'k':'k', 'g':'ɡ', 'p':'p', 'b':'b', 'f':'f', 'v':'v', 'w':'w', 'hh':'h', 'y':'j', 'r':'ɹ', 'dx':'ɾ', 'nx':'ɾ̃', 'tq':'ʔ', 'er':'ɚ', 'em':'m̩'}  # fmt: skip
IPA2BUCKETE = {v: k for k, v in BUCKEYE2IPA.items()}
# 'Vn':'◌̃'


def ipa2buckeye(ipa_string, lang="eng"):
    update_dict_with_tones(BUCKEYE2IPA, IPA2BUCKETE, lang)
    ipa_symbols = string2symbols(ipa_string, IPA2BUCKETE.keys())[0]
    buckeye_symbols = [IPA2BUCKETE[x] for x in ipa_symbols]
    return " ".join(buckeye_symbols)


def buckeye2ipa(buckeye_string, lang="eng"):
    update_dict_with_tones(BUCKEYE2IPA, IPA2BUCKETE, lang)
    if " " in buckeye_string:
        buckeye_symbols = buckeye_string.split()
    else:
        buckeye_symbols = string2symbols(buckeye_string, BUCKEYE2IPA.keys())[0]
    return "".join([BUCKEYE2IPA[x] for x in buckeye_symbols])


#########################################################################
# ARPABET
ARPABET2IPA = {'AA':'ɑ','AE':'æ','AH':'ʌ','AO':'ɔ','IX':'ɨ','AW':'aʊ','AX': 'ə', 'AXR': 'ɚ','AY':'aɪ','EH':'ɛ','ER':'ɝ','EY':'eɪ','IH':'ɪ','IY':'i','OW':'oʊ','OY':'ɔɪ','UH':'ʊ','UW':'u','UX':'ʉ','B':'b','CH':'tʃ','D':'d','DH':'ð','EL':'l̩','EM':'m̩','EN':'n̩','F':'f','G':'ɡ','HH':'h','JH':'dʒ','K':'k','L':'l','M':'m','N':'n','NG':'ŋ','NX':'ɾ̃','P':'p','Q':'ʔ','R':'ɹ','S':'s','SH':'ʃ','T':'t','TH':'θ','V':'v','W':'w','WH':'ʍ','Y':'j','Z':'z','ZH':'ʒ','DX': 'ɾ',}  # fmt: skip
IPA2ARPABET = {v: k for k, v in ARPABET2IPA.items()}


def ipa2arpabet(ipa_string, lang="eng"):
    update_dict_with_tones(ARPABET2IPA, IPA2ARPABET, lang)
    ipa_symbols = string2symbols(ipa_string, IPA2ARPABET.keys())[0]
    arpabet_symbols = [IPA2ARPABET[x] for x in ipa_symbols]
    return " ".join(arpabet_symbols)


def arpabet2ipa(arpabet_string, lang="eng"):
    update_dict_with_tones(ARPABET2IPA, IPA2ARPABET, lang)
    if " " in arpabet_string:
        arpabet_symbols = arpabet_string.split()
    else:
        arpabet_symbols = string2symbols(arpabet_string, ARPABET2IPA.keys())[0]
    return "".join([ARPABET2IPA[x] for x in arpabet_symbols])


#########################################################################
# TIMIT
CLOSURE_INTERVALS = {
    "BCL": ["B"],
    "DCL": ["D", "JH"],
    "GCL": ["G"],
    "PCL": ["P"],
    "TCL": ["T", "CH"],
    "KCL": ["K"],
}
TIMIT2IPA = {'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AX': 'ə', 'AXR': 'ɚ', 'AX-H': 'ə̥', 'AY': 'aɪ', 'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'IH': 'ɪ', 'IY': 'i', 'OW': 'oʊ', 'OY': 'ɔɪ', 'UH': 'ʊ', 'UW': 'u', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EL': 'l̩', 'EM': 'm̩', 'EN': 'n̩', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'P': 'p', 'Q': 'ʔ', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'V': 'v', 'W': 'w', 'WH': 'ʍ', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ', 'DX': 'ɾ', 'ENG': 'ŋ̍', 'EPI': '', 'HV': 'ɦ', 'H#': '', 'IX': 'ɨ', 'NX': 'ɾ̃', 'PAU': '', 'UX': 'ʉ'}  # fmt: skip
IPA2TIMIT = {v: k for k, v in TIMIT2IPA.items()}
INVERSE_CLOSURE_INTERVALS = {v: k for k, val in CLOSURE_INTERVALS.items() for v in val}


def parse_timit(lines):
    # parses the format of a TIMIT .PHN file, handling edge cases where the closure interval and stops are not always paired
    timestamped_phonemes = []
    closure_interval_start = None
    for line in lines:
        if line == "":
            continue
        start, end, phoneme = line.split()
        phoneme = phoneme.upper()

        if closure_interval_start:
            cl_start, cl_end, cl_phoneme = closure_interval_start
            if phoneme not in CLOSURE_INTERVALS[cl_phoneme]:
                ipa_phoneme = TIMIT2IPA[CLOSURE_INTERVALS[cl_phoneme][0]]
                timestamped_phonemes.append((ipa_phoneme, int(cl_start), int(cl_end)))
            else:
                assert phoneme not in CLOSURE_INTERVALS
                start = cl_start

        if phoneme in CLOSURE_INTERVALS:
            closure_interval_start = (start, end, phoneme)
            continue

        ipa_phoneme = TIMIT2IPA[phoneme]
        timestamped_phonemes.append((ipa_phoneme, int(start), int(end)))

        closure_interval_start = None

    if closure_interval_start:
        cl_start, cl_end, cl_phoneme = closure_interval_start
        ipa_phoneme = TIMIT2IPA[CLOSURE_INTERVALS[cl_phoneme][0]]
        timestamped_phonemes.append((ipa_phoneme, int(cl_start), int(cl_end)))

    return timestamped_phonemes


def ipa2timit(ipa_string, lang="eng"):
    update_dict_with_tones(TIMIT2IPA, IPA2TIMIT, lang)
    ipa_symbols = string2symbols(ipa_string, IPA2TIMIT.keys())[0]
    timit_symbols = [IPA2TIMIT[x] for x in ipa_symbols]
    # insert closure intervals before each stop
    timit_symbols_with_closures = []
    for timit_symbol in timit_symbols:
        if timit_symbol in INVERSE_CLOSURE_INTERVALS:
            timit_symbols_with_closures.append(INVERSE_CLOSURE_INTERVALS[timit_symbol])
        timit_symbols_with_closures.append(timit_symbol)
    return " ".join(timit_symbols_with_closures)


def timit2ipa(timit_string, lang="eng"):
    update_dict_with_tones(TIMIT2IPA, IPA2TIMIT, lang)
    if " " in timit_string:
        timit_symbols = timit_string.split()
    else:
        timit_symbols = string2symbols(
            timit_string, TIMIT2IPA.keys() | CLOSURE_INTERVALS.keys()
        )[0]
    timestamped_phonemes = parse_timit((f"0 0 {x}" for x in timit_symbols))
    return "".join([x[0] for x in timestamped_phonemes])


#########################################################################
# CLI
def usage():
    print("Usage: python ./scripts/core/codes.py <src> <tgt> <phoneme_string>")
    print("Supported codes:", CODES)


def main(args):
    if len(args) != 3:
        usage()
        return

    src, tgt, phoneme_string = args
    print(phoneme_string, "=>", convert(phoneme_string, src, tgt))


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"Line {e.__traceback__.tb_lineno}:", e)  # type: ignore
        usage()
