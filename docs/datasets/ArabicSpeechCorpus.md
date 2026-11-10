# Arabic Speech Corpus

Loader `scripts/data_loaders/ArabicSpeechCorpus.py`, class `ArabicSpeechCorpusDataset`. Phone table `scripts/data_loaders/arabic_speech_corpus_phones.json`.

Written Nov 10, 2026. Every number here comes from a script run on the real zip, unless it says it comes from a source. Sources are listed at the end.

## In short

* Modern Standard Arabic, `ISO 639-3` code `arb`, Glottocode `stan1318`, read by one male speaker with a Damascus accent, recorded in a studio for speech synthesis.
* 1,813 train and 100 test utterances in the zip, 4.10 hours of audio. The zip is 1,192,302,846 bytes, with `SHA-256` hash `1df85219370fb1ebe8bfc46aa886265586411d04e7c1caa5a5b9847b3ad5f9de`.
* Every utterance has phone boundaries in a Praat TextGrid.
* The phones were made by a rule based phonetiser from fully diacritised text, aligned by HTK, then corrected by people in part. They say what the speaker was expected to say, with a human check of part of the corpus. They are not a free phonetic transcription.
* One speaker only, so it says nothing about other speakers or accents.
* The license is stated differently by different sources. This needs Alex's call before anything is redistributed.

## The data

| | train | test | all |
| --- | --- | --- | --- |
| Utterances in the zip | 1,813 | 100 | 1,913 |
| Hours of audio | 3.8179 | 0.2860 | 4.1039 |
| Shortest clip, seconds | 1.395 | 3.081 | 1.395 |
| Median clip, seconds | 6.387 | 8.945 | 6.452 |
| 95th percentile, seconds | 16.777 | 19.089 | 17.027 |
| Longest clip, seconds | 36.137 | 28.676 | 36.137 |
| Clips over 30 seconds | 5 | 0 | 5 |
| Items the loader gives | 1,792 | 100 | 1,892 |
| Hours the loader gives | 3.7501 | 0.2860 | 4.0361 |

* Every WAV is 48,000 Hz, mono, 16 bit PCM. The samples start at byte 1024, after a JUNK, bext, fmt and minf chunk and others.
* No two files share audio. No sample reaches full scale. The quietest file still has a 20 ms frame at minus 18.5 dBFS.
* Every WAV has a TextGrid and a line in `orthographic-transcript.txt`, and the other way round.
* Names are `ARA NORM  NNNN` with two spaces. Train runs from 0002 to 1814 with no gap. Test runs from 0001 to 0100. 99 names are in both folders, so the loader's ids carry the split, as in `test/ARA NORM  0002`.
* The phone tier ends 6.7 to 25.0 ms before the audio ends, in every file. There is no gap and no overlap between intervals, and no empty one.
* The thesis gives 3.7 hours and 1,780 utterances in `Table 4-8`. The zip has 1,813 train utterances and 3.82 hours. Why the counts differ is not known.

## How the labels were made

This answers Alex's question on human and machine annotation.

1. The phone sequence came from rules. Section 5.1 of the thesis, on p. 52, says the phonetic transcript was made by rule, as quoted below, with the speaker's dialect taken into account. Halabi has published a phonetiser for diacritised Arabic text. Whether it is the exact version used for this corpus is not verified.

```
done automatically using: classical Arabic orthography rules
```


2. The boundaries came from HTK forced alignment with a flat start, then a second round trained on hand aligned utterances, as sections 5.2 and 6.5 describe. The thesis reports 68.49% of boundaries within 20 ms after the flat start, and 82.50% after the second round.
3. People then corrected part of it. Section 2.3.3, on p. 19, says about 10% of the corpus was aligned by hand after the first attempt. In section 5.2, correcting means moving boundaries and fixing, deleting or adding phone labels. Two experts aligned, and a third reviewed where they disagreed, which section 6.4 puts at 10% to 20% of the corpus.
4. Stress marks came from orthography based rules by Halpern, 2009, as the thesis and the Wikipedia article on the corpus describe.

So the labels are rule based and machine aligned, with a human pass over a part. The share that people checked differs by source. The thesis gives about 10% aligned by hand. The Hugging Face card says the whole corpus was checked, in these words. Which is right is not verified.

```
The entire corpus was checked by human annotators.
```

## The annotators

* `Table 2-1` of the thesis, in section 2.4 on p. 22, names two experts. Expert 1 was 51, with a PhD in linguistics and grammar. Expert 2 was 31, with a BA in English to Arabic translation. A third expert reviewed the disagreements, as section 6.4 says.
* Section 5.4 says the correctors had three 3 hour training sessions before they started.
* The Hugging Face card lists Nawar Halabi and two anonymous Arabic language teachers.
* In section 6.4, the two experts placed 84.28% of 884 boundaries within 20 ms of each other. The thesis gives 93.49% for TIMIT for comparison. This relates to issue #10. The 884 matches the count of normal utterances in `Table 4-8`, so it should be checked on the page that it counts boundaries.

## The phone symbols

The TextGrids use Halabi's own symbols, a modified Buckwalter transliteration. The loader maps them to IPA with the table in `arabic_speech_corpus_phones.json`, which names a source for every entry.

* Consonants follow Buckwalter letters, such as `E` for ʕ, `H` for ħ, `$` for ʃ, `^` for θ, `*` for ð, `g` for ɣ and `<` for ʔ. The emphatics are `S`, `D`, `T` and `Z`, mapped to sˤ, dˤ, tˤ and ðˤ.
* A doubled consonant symbol is a geminate, such as `dd`. The loader writes it with the length mark, as dː.
* Vowels are `a`, `i0`, `i1`, `u0` and `u1`, with `aa`, `ii0`, `uu0` and others for long ones. A 1 marks a vowel that leans toward a schwa, and a 0 one that does not, as section 5.1 of the thesis says. Capital letters, such as `A` and `I0`, mark a vowel next to an emphatic consonant.
* An apostrophe after a vowel marks a stressed syllable. A syllable of Modern Standard Arabic starts with one consonant, as Broselow describes, so the loader writes ˈ before the consonant right before that vowel, as in ˈba. Before a geminate it goes before the whole geminate. The timestamps carry no stress, so each phone unit stays a plain phone. So `split_utterance_into_multiple`, which joins the timestamps, gives IPA with no stress, and `vocab` has no ˈ.
* `sil` is a pause. It gives an empty phone in the timestamps and nothing in the IPA.
* So the labels are phonemic with some allophones. Leaning, emphasis next to a consonant and stress come from rules, not from listening.

Four points need Alex's check, since they are calls of mine or rest on a table the PDF's text does not show cleanly.

1. `u0` is mapped to u. The thesis table, as its text came out of the PDF, prints æ on the u0 row, but the same row's Arabic letter is the damma, and `uu0` is uː. This is 4,353 intervals.
2. Capital `I` and `U` vowels get the IPA of the lower case vowel, since the thesis gives them no IPA of their own. This is 3,597 intervals. Their backing is lost from the IPA. Next to sˤ or tˤ the consonant still shows it, but not always. In item 0058, `U0` sits between `t` and `l`. `A` and `AA` do keep it, as ɑ and ɑː.
3. `j` is ʒ and `J` is d͡ʒ, from the LREC 2016 paper's table, since the thesis table lists both for that letter.
4. `a` is a but `aa` is æː in the extracted table, while every other short and long pair keeps one quality. The extraction may have shifted a column, and then `a` would be æ. This is 28,660 intervals of `a`, `a'`, `aa` and `aa'`. Every vowel row on p. 46 should be checked by eye.

The original labels stay in `raw_label` in the speaker info, so a different mapping can be built later without the zip.

## Cleaning

Done by the loader, with counts from the real run.

* An utterance is skipped when one of its phones has no IPA. These are `dist`, which the README says marks corrupt phonemes, `G`, `AH`, `Ah` and a bare `u`. The table says why for each. They are 27 intervals of 139,085, and they skip 21 utterances, all in train. By the first such phone in each, that is 10 for `dist`, 4 for `Ah`, 3 for `G`, 3 for `u` and 1 for `AH`.
* 3 labels carry a trailing space, such as `E `. The space is removed and counted.
* An utterance with no phone at all would be skipped and counted. The corpus has none.
* A label in neither list stops the loader with an error that names the file and says to add the label to the table.

Still to do.

* Nonsense utterances. `Table 4-8` of the thesis gives 896 utterances built from 4 templates to cover missing diphones. The corpus marks them nowhere, since every file is `ARA NORM`. Two template patterns in the text find 672 of them, 1.11 hours. The loader keeps them all. Alex may want them out of a leaderboard.
* Nobody has listened to the clips. The suite writes 6 for listening to `tmp/listen/arabic/`, outside the PR.
* The IPA of the vowels rests on one table that the PDF's text does not show cleanly. A person should check page 46 of the thesis against the JSON.
* The README says the test set was recorded at another time, so it may sound a little different from train.
* Do not build labels from `phonetic-transcript.txt`. It differs from the TextGrids in 11 utterances and writes `TH` and `SH` in places. The train copy of that file is spelled `phonetic-transcipt.txt`. The lab files also write `TH` for `*`.

## What evaluation drops

As a crash test, `evaluate(ipa, ipa)` ran on all 1,891 distinct labels, and gave PER 0 and FER 0 on each. The useful part is what `preprocess_ipa` does. Of the 73 phone units and the stress mark, it changes 2.

* It drops every stress mark, 16,006 of them, since panphon does not know ˈ.
* It drops the length of o̞ː, 33 times, so o̞ː is scored as o̞.
* Every other unit passes unchanged, the geminates, the emphatics and the other long vowels included.

## Splits

* `train` and `test` are the corpus's own folders. `all` is train, then test. Items keep file name order, so every machine gets the same order.
* Section 7.4 of the thesis says the 100 test sentences were picked by hand from 135 of the same text source. No test sentence is also a train sentence, and no test utterance shares a 5 word run with train.
* The default split is `train`.

## License

The sources do not agree.

* The README in the zip says commercial use needs an agreement with the owner, MicroLinkPC UK Limited, and that research use is free, in these words.

```
License: Commercial (Commercial use only allowed after agreement with OWNER).
The corpus is free for research purposes.
```

* The ELRA catalogue, item `ELRA-S0384`, offers a Creative Commons Attribution License, or a paid ELRA agreement for commercial use.
* The Oxford Text Archive record 2561 gives `CC BY 4.0`. The Oxford LLDS copy of the same record gives `CC BY-NC-SA 3.0`.
* ISLRN and Wikipedia give `CC BY-NC-SA 4.0`. The Hugging Face card gives `CC BY 4.0`.
* The official site could not be read from this session. Raunak's download notes record `CC BY 4.0` from it.

The loader reads a local copy only. Whether Koel can redistribute the audio or the labels is Alex's call.

## Use

* Training, with care. It is one speaker, and its labels are rule based. If Koel adopts the `label_source` field proposed in the DoReCo notes, this corpus would be `g2p_forced_aligned_human_corrected`, with the correction partial.
* Leaderboard, only as a narrow test. A model would be scored on one voice against rule based labels.
* Not covered are other speakers, women, children, other accents and spontaneous speech.

## Load time and memory

On the VM, Linux with 4 CPUs, in two runs of 3 setups each, setting up `train` took 0.94 s and 0.97 s at best. Reading items 0 to 49 ran at 15.7 to 17.9 items per second, since each item reopens the zip and resamples 48 kHz audio. Peak memory was 304 and 305 MB. A pass over all 1,892 items with every output took about 100 s. The Mac was not measured.

## Sources

Read on Nov 10, 2026. The thesis and the LREC paper were read through a text extraction of the PDF, not on the rendered page.

* Halabi, N., 2016. Modern Standard Arabic phonetics for speech synthesis. PhD thesis, University of Southampton. https://eprints.soton.ac.uk/409695/
* Halabi, N. and Wald, M., 2016. Phonetic inventory for an Arabic speech corpus. LREC 2016, pp. 734 to 738. https://eprints.soton.ac.uk/397310/
* README.txt in `arabic-speech-corpus.zip`, version 2.0
* Hugging Face card, https://huggingface.co/datasets/arabic_speech_corpus
* ELRA catalogue, https://catalog.elra.info/en-us/repository/browse/ELRA-S0384
* Oxford Text Archive, https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2561
* Oxford LLDS, https://llds.phon.ox.ac.uk/llds/xmlui/handle/20.500.14106/2561
* ISLRN, https://islrn.org/resources/866-568-447-697-8
* Halabi's phonetiser, https://github.com/nawarhalabi/Arabic-Phonetiser
* Wikipedia, Arabic Speech Corpus, https://en.wikipedia.org/wiki/Arabic_Speech_Corpus
* Broselow, E., Arabic syllable structure, https://linguistics.stonybrook.edu/faculty/ellen.broselow/files/Broselow_ArabicSyllStrPaper_revised.pdf
* Glottolog, Standard Arabic, https://glottolog.org/resource/languoid/id/stan1318
* The site text on stress marks, as copied at https://dagshub.com/DagsHub/audio-datasets/src/main/Arabic-Speech-Corpus
