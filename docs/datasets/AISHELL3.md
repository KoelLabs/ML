# AISHELL-3

Loader `scripts/data_loaders/AISHELL3.py`, class `AISHELL3Dataset`. Pinyin to IPA table `scripts/data_loaders/aishell3_pinyin_ipa.json`.

Written Nov 10, 2026, Pacific time. Every number here comes from a script run on the real archive, unless it says it comes from a source. Sources are listed at the end.

## In short

* Mandarin Chinese, `ISO 639-3` code `cmn`, Glottocode `mand1415`, read from scripts by 218 native speakers, recorded for speech synthesis.
* 88,035 utterances, 85.62 hours of audio. The archive is 19,057,141,777 bytes, with `SHA-256` hash `be2507d431ad59419ec871e60674caedb2b585f84ffa01fe359784686db0e0cc`.
* People wrote the characters and the pinyin with tones by listening to each recording. The IPA is not an annotation. The loader makes it from the pinyin with a fixed table, so it is phonemic, and it shows no allophones and no accent beyond what the pinyin writes.
* There are no phone times, so the loader gives no timestamps.
* 176 women and 42 men. Women hold 81.0% of the hours the loader gives.
* The license is stated differently by different sources. This needs Alex's call before anything is redistributed.

## The data

| | train | test | `test_new_speakers` | all |
| --- | --- | --- | --- | --- |
| Utterances in the archive | 63,262 | 24,773 | 13,707 | 88,035 |
| Hours of audio | 63.17 | 22.45 | 12.91 | 85.62 |
| Items the loader gives | 62,033 | 24,246 | 13,404 | 86,279 |
| Hours the loader gives | 61.89 | 21.97 | 12.62 | 83.86 |
| Speakers | 174 | 214 | 44 | 218 |
| Women, men | 138, 36 | 172, 42 | 38, 6 | 176, 42 |
| Shortest clip, seconds | 1.60 | 0.79 | 0.84 | 0.79 |
| Median clip, seconds | 3.30 | 2.99 | 3.10 | 3.21 |
| 95th percentile, seconds | 6.32 | 5.94 | 5.98 | 6.22 |
| Longest clip, seconds | 11.01 | 15.72 | 10.68 | 15.72 |

* Every WAV is 44,100 Hz, mono, 16 bit PCM. In 80,677 files the RIFF size field is larger than the file, by one of 6 fixed amounts. In all 88,035 the data chunk ends exactly at the end of the file. So the loader checks the data chunk instead. A cut file still fails, and a chunk after the data is allowed.
* No two files share audio. Every WAV has a line in `content.txt`, and the other way round. 67 files have a sample at full scale, at most 747 such samples in one file.
* Speaker SSB0987 is quiet. The median of its loudest 20 ms frames is minus 34.2 dBFS, against minus 20.45 for the median speaker. 17 files have no frame above minus 40 dBFS, 16 of them from this speaker. None is silent, since every file has a frame above minus 60 dBFS. Nobody has listened to them yet.
* Age groups, from `spk-info.txt`, are 2 under 14, 173 from 14 to 25, 35 from 26 to 40, and 8 over 41. Accents are 165 north, 51 south and 2 others. No man has the accent others.
* `ReadMe.txt` says 43 men and 175 women. `spk-info.txt` and Table 2 of the paper both give 42 men and 176 women. One gender in `spk-info.txt` is written with a trailing space, and the loader strips it.
* The paper gives 64,773 utterances for its train set. The train folder has 63,262 lines. Why they differ is not known.

## How the labels were made

This answers Alex's question on human and machine annotation.

1. The speakers read from given scripts with neutral emotion, as section 2 of the paper says.
2. People then wrote the pinyin, tones included, from what each speaker actually said. Section 2 of the paper says so, in these words.

```
The pinyin transcripts are obtained through human listening tests and directly corresponds to the speaker's actual readings.
```

3. So the pinyin follows the speech, not a dictionary. Third tone sandhi is written as spoken, as in `ni2 you3` in SSB07840471, where a dictionary gives ni3. Section 2 of the paper names four reasons why pinyin cannot come from the characters alone, which are words with two readings, tone sandhi, erhua, and accents or misreadings.
4. The OpenSLR page and the Hugging Face card of the publisher give an accuracy figure, in these words. Neither says how it was measured.

```
The word & tone transcription accuracy rate is above 98%, through professional speech annotation and strict quality inspection for tone and prosody.
```

5. The train folder also has a prosody label file. Its pinyin equals `content.txt` in 63,260 of its 63,262 lines.
6. The IPA was made by this loader, not by the publisher. Each pinyin syllable becomes IPA through the table, as the next part shows.

So the characters and the pinyin are human labels, checked by people. The IPA is a fixed rule over the human pinyin. If Koel adopts the `label_source` field proposed in the DoReCo notes, none of its values fits, and a new one such as `human_pinyin_table` would be needed. That is Alex's call.

## The annotators

* The paper does not say who wrote the pinyin, how many people did, or how they were trained.
* The publisher's pages say only that it was professional speech annotation with strict quality inspection, as quoted above.
* No agreement between annotators is reported. This relates to issue #10.

## The phone symbols

The loader writes 34 phone units, and 42 with the tone letters.

```
a e f i k kʰ l m n o p pʰ s t ts tsʰ tɕ tɕʰ tʂ tʂʰ tʰ u x y ŋ ɔ ɕ ə ɛ ɤ ɹ ɹ̩ ɻ̩ ʂ
˥ ˧˥ ˨˩˦ ˥˩ ˨ ˧ ˦ ˩
```

* Initials and finals follow the IPA pages of the Chinese 342 course at the University of Washington. They are broad, and write the glides as i, u and y. The pages were read through a text extraction, and the consonant table came out flattened, so a few pairings follow the order of its cells and are not checked on the rendered page.
* The vowel of zi, ci, si is ɹ̩, and the vowel of zhi, chi, shi, ri is ɻ̩, from a Wikibooks page read in a search extract. It gives the two symbols, but not which goes with which group, so that pairing is not verified. The usual signs ɿ and ʅ are not in the repo's symbol set, so the table uses the other spelling.
* The initial r is ɹ, as in the University of Washington table. Other sources write ɻ or ʐ. So ri comes out as ɹɻ̩. This is one line in the JSON, for Alex.
* The final iong is yuŋ, as in the same table. Other broad tables write iʊŋ or jʊŋ. Also one line, for Alex.
* An affricate is one unit, such as tʂ, written without a tie bar.
* Tones use Chao tone letters, from Wikipedia, read in a search extract. Tone 1 is ˥, tone 2 ˧˥, tone 3 ˨˩˦, tone 4 ˥˩.
* The neutral tone 5 takes its pitch from the full tone before it, from a LibreTexts course page read in a search extract. It is ˨ after tone 1, ˧ after tone 2, ˦ after tone 3 and ˩ after tone 4. The source does not say what a run of neutral tones does, or one at the start of an utterance. The loader keeps the pitch of the last full tone for a run, and writes ˧ at the start. Those two choices are the loader's own, not verified.
* Tone 3 is always written in full, though it is often spoken low only. Only the sandhi that the pinyin writes is shown.
* Each choice and its source is in the JSON, so a different reading changes one line.

## Cleaning

Done in this loader.

* 1,756 utterances are skipped and counted, 2.0% of the archive. 1,749 have erhua, a syllable ending in r other than er, such as `dianr3`. The vowel of an erhua syllable changes in a way that a table per syllable cannot give, so any rule would be a guess. 2 have `yo`, which the table does not know. 5 have a syllable that the table lists to skip, 3 `lo`, a particle, since the table reads o as uɔ, which holds only after b, p, m and f, 1 `jv`, a nonstandard spelling of ju, and 1 `din`, which is not a pinyin syllable.
* So 172 of the 174 train speakers and 213 of the 214 test speakers keep items. The two train speakers and the one test speaker who lose all their items, SSB1377, SSB1630 and SSB0629, have one line each in that folder.
* The files have Windows line ends, and 11 lines end in a space. The loader reads both.
* A line of `content.txt` that the loader cannot read, or a file name that is there twice, stops the load with an error that names it. Nothing is guessed.
* The loader turns 44.1 kHz into 16 kHz with the repo's shared resampler, as every loader does. It uses linear steps with no low pass filter, so some aliasing is likely. This was not measured here.

Still to do.

* Erhua, 2.0% of the data, needs a rule written by someone who knows Mandarin phonology, or it stays out.
* Nobody has listened to the clips. 7 clips for listening are kept outside the PR, the quietest file among them.
* A person who knows Mandarin should check the table, mainly r, iong, the two buzzed vowels and the neutral tone.
* Accent is in the speaker info, but the labels do not show it.

## What evaluation drops

As a crash test, `evaluate(ipa, ipa)` ran on all 83,504 distinct labels without tones, and gave PER 0 and FER 0 on each. On a sample of the first 6,000 of the 84,025 distinct labels with tones, it gave the same. The useful part is what `preprocess_ipa` and the grouping do.

* It drops every tone letter. So with tones, FER is exactly the same as without, and a model that gets the tones right scores no better on FER.
* It adds a tie bar to ts, tsʰ, tɕ and tɕʰ, and scores each as one phone.
* It splits tʂ into t and ʂ, and tʂʰ into t and ʂʰ. So the retroflex affricates count as two phones each, while the other affricates count as one. This relates to issue #15.
* Every other unit passes unchanged.

## Splits

* `train` and `test` are the archive's own folders. `all` is train, then test. Items keep file order, so every machine gets the same order.
* The test folder is not new speakers only. 170 of its 214 speakers are also in train, with 11,066 of its 24,773 lines. Section 4.1.2 of the paper explains this. 44 speakers were drawn at random as unseen test speakers, and part of each train speaker's lines were kept back for checking on seen speakers.
* `test_new_speakers` is the test lines of the speakers who are not in train. These are the paper's unseen speakers, 44 speakers with 13,707 lines, of which the loader gives 13,404 items. Use it for a leaderboard.
* The default split is `train`.

## License

The sources do not agree.

* OpenSLR gives `Apache License v.2.0`, and the paper's abstract says the corpus is available under Apache v2.0.
* The Hugging Face card of the publisher gives `apache-2.0`.
* `ReadMe.txt` in the archive has no license line.
* The VoxKitchen dataset catalog gives `CC BY-NC-ND 4.0`.
* A post from the company on CSDN, seen on Nov 10, said the corpus is free for academic research and not for commercial use without permission. Its address was not kept, so this is not verified.

The loader reads a local copy only. Whether Koel can redistribute the audio or the labels is Alex's call.

## Use

* Training, for a phonemic model of read Mandarin. 218 speakers is many, though most are women from 14 to 25 with a northern accent.
* Leaderboard, on `test_new_speakers`, if the labels are named as table labels. A model would be scored against a table reading of the pinyin, not against a phonetic transcription.
* Not covered are spontaneous speech, children, older speakers, accents other than north and south, and men with the accent others.

## Load time and memory

On the VM, Linux with 4 CPUs, setting up took 2.50 s for `train`, 0.96 s for `test` and 3.68 s for `all`, at best of 3. Reading 50 items spread over `all` ran at 332.1 items per second, then 523.2 on a second pass. Peak memory was 509 MB. The Mac was not measured.

## Sources

Read on Nov 10, 2026, Pacific time. The paper was read through text extractions of the arXiv HTML and PDF, not on the rendered page.

* Shi, Y., Bu, H., Xu, X., Zhang, S. and Li, M., 2021. AISHELL-3, a multi speaker Mandarin TTS corpus. Interspeech 2021, pp. 2756 to 2760. https://www.isca-archive.org/interspeech_2021/shi21c_interspeech.html
* The same paper on arXiv, 2010.11567 version 2, https://arxiv.org/abs/2010.11567, and its HTML, https://ar5iv.labs.arxiv.org/html/2010.11567
* `ReadMe.txt`, `spk-info.txt`, `content.txt` and `ChangeLog` in `data_aishell3.tgz`
* OpenSLR, SLR93, https://www.openslr.org/93/
* Hugging Face card of the publisher, https://huggingface.co/datasets/AISHELL/AISHELL-3
* VoxKitchen dataset catalog, https://voxkitchen.readthedocs.io/en/latest/datasets/aishell3/
* University of Washington, Chinese 342 IPA pages, finals and consonants, the source of the table, linked in the JSON
* Wikibooks and Wikipedia pages on Mandarin phonology and tone letters, linked in the JSON
* Wikipedia, Mandarin Chinese, for the `ISO 639-3` code and the Glottocode, https://en.wikipedia.org/wiki/Mandarin_Chinese
* LibreTexts, CHN 101 Beginning Mandarin, 1.2.1 Mandarin Phonology II, section Neutral Tone, the source of the neutral tone, linked in the JSON
