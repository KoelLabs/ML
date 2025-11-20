#!/usr/bin/env python3

# utils for processing text

import os
import sys
import re
import json
import unicodedata
from fractions import Fraction

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.codes import arpabet2ipa
from core.ipa import filter_chars

from string import punctuation
import g2p_en

g2p = g2p_en.G2p()


def english2ipa(text, filter_type="letters_rmv_tie"):
    arpa = [g2p(line) for line in text.split("\n")]

    def arpa2ipa(a):
        if a == " ":
            return " "
        try:
            return arpabet2ipa(a)
        except:
            return a

    text = remove_punctuation("\n".join("".join(map(arpa2ipa, arp)) for arp in arpa))
    return filter_chars(text, filter_type)


def remove_punctuation(text):
    return "".join([c for c in text if c not in punctuation])


# from https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py
def normalize_english_numbers(text: str):
    """Convert any spelled-out numbers into arabic numbers, remove any commas, keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.,
    spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`, spell out `one` and `ones`,
    interpret successive single-digit numbers as nominal: `one oh one` -> `101`"""

    ZEROS = {"o", "oh", "zero"}
    ONES = {name: i for i, name in enumerate(["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",],start=1,)}  # fmt: skip
    ONES_SUFFIXED = {**{"sixes" if name == "six" else name + "s": (value, "s")for name, value in ONES.items()}, **{"zeroth": (0, "th"),"first": (1, "st"),"second": (2, "nd"),"third": (3, "rd"),"fifth": (5, "th"),"twelfth": (12, "th"), **{name + ("h" if name.endswith("t") else "th"): (value, "th")for name, value in ONES.items()if value > 3 and value != 5 and value != 12},},}  # fmt: skip
    TENS = {"twenty": 20,"thirty": 30,"forty": 40,"fifty": 50,"sixty": 60,"seventy": 70,"eighty": 80,"ninety": 90}  # fmt: skip
    TENS_SUFFIXED = {**{name.replace("y", "ies"): (value, "s") for name, value in TENS.items()}, **{name.replace("y", "ieth"): (value, "th") for name, value in TENS.items()}}  # fmt: skip
    DECIMALS = {*ZEROS, *ONES, *TENS}
    MULTIPLIERS = {"hundred": 100,"thousand": 1_000,"million": 1_000_000,"billion": 1_000_000_000,"trillion": 1_000_000_000_000,"quadrillion": 1_000_000_000_000_000,"quintillion": 1_000_000_000_000_000_000,"sextillion": 1_000_000_000_000_000_000_000,"septillion": 1_000_000_000_000_000_000_000_000,"octillion": 1_000_000_000_000_000_000_000_000_000,"nonillion": 1_000_000_000_000_000_000_000_000_000_000,"decillion": 1_000_000_000_000_000_000_000_000_000_000_000,}  # fmt: skip
    MULTIPLIERS_SUFFIXED = {**{name + "s": (value, "s") for name, value in MULTIPLIERS.items()}, **{name + "th": (value, "th") for name, value in MULTIPLIERS.items()}}  # fmt: skip

    PRECEDING_PREFIXES = {"minus": "-","negative": "-","plus": "+","positive": "+"}  # fmt: skip
    FOLLOWING_PREFIXES = {"pound": "£","pounds": "£","euro": "€","euros": "€","dollar": "$","dollars": "$","cent": "¢","cents": "¢"}  # fmt: skip
    PREFIXES = set(list(PRECEDING_PREFIXES.values()) + list(FOLLOWING_PREFIXES.values()))  # fmt: skip
    SUFFIXES = {"per": {"cent": "%"},"percent": "%"}  # fmt: skip
    SPECIALS = {"and", "double", "triple", "point"}
    WORDS = {key for mapping in [ZEROS,ONES,ONES_SUFFIXED,TENS,TENS_SUFFIXED,MULTIPLIERS,MULTIPLIERS_SUFFIXED,PRECEDING_PREFIXES,FOLLOWING_PREFIXES,SUFFIXES,SPECIALS, ] for key in mapping}  # fmt: skip

    # replace "<number> and a half" with "<number> point five"
    results = []
    segments = re.split(r"\band\s+a\s+half\b", text)
    for i, segment in enumerate(segments):
        if len(segment.strip()) == 0:
            continue
        if i == len(segments) - 1:
            results.append(segment)
        else:
            results.append(segment)
            last_word = segment.rsplit(maxsplit=2)[-1]
            if last_word in DECIMALS or last_word in MULTIPLIERS:
                results.append("point five")
            else:
                results.append("and a half")
    text = " ".join(results)

    # put a space at number/letter boundary
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"([0-9])([a-z])", r"\1 \2", text)

    # but remove spaces which could be a suffix
    text = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", text)

    def process_words(words: list[str]):
        prefix: "str | None" = None
        value: "str | int | None" = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in PREFIXES
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in WORDS:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in ZEROS:
                value = str(value or "") + "0"
            elif current in ONES:
                ones = ONES[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in ONES:
                    if (
                        prev in TENS and ones < 10
                    ):  # replace the last zero with the digit
                        value = value[:-1] + str(ones)  # type: ignore
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in ONES_SUFFIXED:
                # ordinal or cardinal; yield the number right away
                ones, suffix = ONES_SUFFIXED[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in ONES:
                    if prev in TENS and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)  # type: ignore
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in TENS:
                tens = TENS[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in TENS_SUFFIXED:
                # ordinal or cardinal; yield the number right away
                tens, suffix = TENS_SUFFIXED[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in MULTIPLIERS:
                multiplier = MULTIPLIERS[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)  # type: ignore
                    p = f * multiplier if f is not None else None
                    if p is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in MULTIPLIERS_SUFFIXED:
                multiplier, suffix = MULTIPLIERS_SUFFIXED[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if p is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in PRECEDING_PREFIXES:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in WORDS or next_is_numeric:
                    prefix = PRECEDING_PREFIXES[current]
                else:
                    yield output(current)
            elif current in FOLLOWING_PREFIXES:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = FOLLOWING_PREFIXES[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in SUFFIXES:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = SUFFIXES[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in SPECIALS:
                if next not in WORDS and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in MULTIPLIERS:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in ONES or next in ZEROS:
                        repeats = 2 if current == "double" else 3
                        ones = ONES.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in DECIMALS or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    text = " ".join(word for word in process_words(text.split()) if word is not None)

    # normalize currencies: "$2 and ¢7" -> "$2.07"
    def combine_cents(m):
        try:
            currency = m.group(1)
            integer = m.group(2)
            cents = int(m.group(3))
            return f"{currency}{integer}.{cents:02d}"
        except ValueError:
            return m.string

    def extract_cents(m):
        try:
            return f"¢{int(m.group(1))}"
        except ValueError:
            return m.string

    text = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, text)
    text = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, text)

    # write "one(s)" instead of "1(s)", just for the readability
    text = re.sub(r"\b1(s?)\b", r"one\1", text)

    return text


with open(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "english-spelling-variants.json"
    ),
    "r",
) as f:
    SPELLING_VARIANTS = json.load(f)


# from https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py
def normalize_english(text: str):
    text = text.lower()

    text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)  # remove words between brackets
    text = re.sub(r"\(([^)]+?)\)", "", text)  # remove words between parenthesis
    text = re.sub(r"\b(hmm|mm|mhm|mmm|uh|um)\b", "", text)  # remove uh and mms

    # standardize when there's a space before an apostrophe
    text = re.sub(r"\s+'", "'", text)

    # standardize contractions
    CONTRACTIONS = {r"\bwon't\b": "will not",r"\bcan't\b": "can not",r"\blet's\b": "let us",r"\bain't\b": "aint",r"\by'all\b": "you all",r"\bwanna\b": "want to",r"\bgotta\b": "got to",r"\bgonna\b": "going to",r"\bi'ma\b": "i am going to",r"\bimma\b": "i am going to",r"\bwoulda\b": "would have",r"\bcoulda\b": "could have",r"\bshoulda\b": "should have",r"\bma'am\b": "madam",r"\bmr\b": "mister ",r"\bmrs\b": "missus ",r"\bst\b": "saint ",r"\bdr\b": "doctor ",r"\bprof\b": "professor ",r"\bcapt\b": "captain ",r"\bgov\b": "governor ",r"\bald\b": "alderman ",r"\bgen\b": "general ",r"\bsen\b": "senator ",r"\brep\b": "representative ",r"\bpres\b": "president ",r"\brev\b": "reverend ",r"\bhon\b": "honorable ",r"\basst\b": "assistant ",r"\bassoc\b": "associate ",r"\blt\b": "lieutenant ",r"\bcol\b": "colonel ",r"\bjr\b": "junior ",r"\bsr\b": "senior ",r"\besq\b": "esquire ",r"'d been\b": " had been",r"'s been\b": " has been",r"'d gone\b": " had gone",r"'s gone\b": " has gone",r"'d done\b": " had done",r"'s got\b": " has got",r"n't\b": " not",r"'re\b": " are",r"'s\b": " is",r"'d\b": " would",r"'ll\b": " will",r"'t\b": " not",r"'ve\b": " have",r"'m\b": " am"}  # fmt: skip
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # remove commas between digits and periods not followed by numbers
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"\.([^0-9]|$)", r" \1", text)

    # normalize unicode, keep some symbols for numerics, otherwise replace markers, symbols, and punctuations with a space (MSP) and drop diacritics (Mn + manual mappings)
    ADDITIONAL_DIACRITICS = {"œ": "oe","Œ": "OE","ø": "o","Ø": "O","æ": "ae","Æ": "AE","ß": "ss","ẞ": "SS","đ": "d","Đ": "D","ð": "d","Ð": "D","þ": "th","Þ": "th","ł": "l","Ł": "L"}  # fmt: skip
    text = "".join(
        (
            c
            if c in ".%$¢€£"
            else ADDITIONAL_DIACRITICS.get(
                c,
                (
                    ""
                    if unicodedata.category(c) == "Mn"
                    else " " if unicodedata.category(c)[0] in "MSP" else c
                ),
            )
        )
        for c in unicodedata.normalize("NFKD", text)
    )

    text = normalize_english_numbers(text)
    text = " ".join(SPELLING_VARIANTS.get(word, word) for word in text.split())

    # now remove prefix/suffix symbols that are not preceded/followed by numbers
    text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
    text = re.sub(r"([^0-9])%", r"\1 ", text)

    # replace any successive whitespace characters with a space
    text = re.sub(r"\s+", " ", text)

    return text


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
    elif command == "normalize_english":
        print(normalize_english(text))
    else:
        print("Unknown command")


if __name__ == "__main__":
    main(sys.argv[1:])
