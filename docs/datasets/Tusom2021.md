# Tusom2021

These notes are from Nov 9, 2026. Every number comes from the real data, the repo https://github.com/dmort27/tusom2021 at commit `e168dfb`, unless a line says otherwise. Facts from the paper give the page of the Interspeech 2021 PDF. Where the paper and the data disagree, both are given. What we could not check says not verified.

## Words used below

* A phone is one speech sound.
* A token is one spoken instance of a word or short phrase, in one WAV file. A type is one word or phrase. Most types here have three tokens.
* A label is the transcription of one token. `mapping.yml` gives each token three labels, one in each form.
* FER and PER are the two scores that the `evaluate` function in `scripts/eval/evaluate.py` returns. Both are error rates, so 0 is a perfect match. PER counts the character edits between the two raw strings. FER weighs the edits by sound features, after `preprocess_ipa` cleans both strings.
* A glottocode is a language id from Glottolog, a catalog of the world's languages.

## Loader

`scripts/data_loaders/Tusom2021.py`, class `Tusom2021Dataset`.

* `split` is `train`, `dev`, `test` or `all`. The first three are the dataset's own files.
* `tones` is `none`, `diacritics` or `letters`, for the fields `no_tones`, `tone_dias` and `tone_letters`. The default is `none`.
* `include_timestamps=True` fails, since the data has no phone times.
* `include_speaker_info=True` gives a dict with `id`, `recording`, `speaker`, `language`, `source` and `raw_label`. `raw_label` is the exact field from `mapping.yml`.
* `include_text=True` gives the English gloss, since East Tusom has no writing system, page 3660.

Why the default is `none`. Both experiments in the paper use that form, on pages 3662 and 3663. Tones also add nothing to FER today, since `preprocess_ipa` drops every tone mark. See Symbols the evaluation drops below. There is one cost. With this form, 10 labels are in two splits, as Splits below says. The default is one argument to change.

## Report

| Item | Value | Source |
|---|---|---|
| Language | East Tusom, a Tangkhulic language of Manipur, India | Paper, page 3660 |
| Language codes | no `ISO 639-3` code, glottocode `east2902` | Paper, page 3660, and the Glottolog entry for Mortensen and Picone 2021 |
| Speakers | 1 | Paper, page 3661 |
| WAV files on disk | 2,392 | Data |
| Labels in `mapping.yml` | 2,255 | Data |
| Items, train, dev, test | 1,578, 230, 447 | Data, equal to Table 5 of the paper |
| WAV files with no label, skipped | 137 | Data |
| Audio minutes, train, dev, test, all | 36.49, 5.20, 10.13, 51.83 | Data, equal to Table 5 to one decimal |
| Clip seconds, all items | shortest 0.223, median 1.303, 95th percentile 2.121, longest 4.772 | Loader |
| Clips over 30 s | 0 | Loader |
| Phone times | none in the data | Data |
| Distinct panphon units | 59 with no tones, 61 with tone marks, 64 with tone letters | Loader |
| WAV format | all 2,392 are mono, 16 bit PCM, plain 44 byte header | Data |
| Sample rate | 2,173 files at 22,050 Hz, 219 at 16,000 Hz | Data |
| Load time, one pass over all 2,255 items | 3.6 s, 625 items per second, on the Linux VM from the Mac's shared folder | `Tusom2021Dataset` on Nov 9 |
| Setup time for all | 1.70 s, best of 3 | Same |
| Peak memory | 338 MB, the whole process with its imports | Same |

Clip seconds per split.

| Split | Shortest | Median | 95th percentile | Longest |
|---|---|---|---|---|
| train | 0.223 | 1.302 | 2.148 | 4.772 |
| dev | 0.682 | 1.321 | 1.987 | 2.863 |
| test | 0.588 | 1.285 | 2.124 | 3.557 |

## Speakers

There is one speaker. The paper says this on page 3661.

```
The data were collected from one native female speaker of East Tusom in her late 20s. She did not grow up in Tusom village but spoke the language at home, with her family, and when she stayed with her extended family in Tusom.
```

The speaker has no code in the data. The loader gives every item the same `speaker` value. With one speaker, this data can say nothing about how a model handles different speakers.

## Speech type and recording

* Elicited words and short phrases from a comparative word list of about 2,400 items, page 3662.
* The first author gave an English word and the speaker answered in East Tusom, three times, page 3662.
* Recorded in 2004, page 3661, on an inexpensive lavalier microphone into the analog jack of a laptop running Audacity, page 3662.
* The paper describes the noise on page 3660. These are its words.

```
The recording equipment was optimized for price and portability, rather than acoustic fidelity and there is considerable background noise (from air conditioning and small children).
```

* The WAV names have 42 stems with labeled items, such as `animals1` or `verb14`. The paper says each tranche of words was recorded in one session, page 3662. That the stems are those sessions is likely, not verified. The loader returns the stem as `recording`.
* Loudness differs much between stems. In 12 of the 42, the median clip is under minus 40 dB, and they hold 665 of the 2,255 items. They are `bp1` to `bp10`, `numbers` and `numbers2`. The median of their medians is minus 44.2 dB, and of the other stems minus 30.5 dB.
* No clip is all zeros. In 38 clips, the loudest 20 ms frame is under minus 40 dB. This check can find silence and clipping. It cannot show that a clip holds the word in its label.

## License and how to get the data

The data repo has an MIT License, copyright 2021 David R. Mortensen, in its `LICENSE` file. Nobody has to sign anything to download it. The commands are in the header of the loader.

The MIT License is written for software. Neither the paper nor the repo says whether it is meant to cover the recordings, or what the speaker agreed to. We make no legal judgment on that.

## Citation

The project page at https://llab-cmu.github.io/projects/tusom-project.html asks users to cite the Interspeech version, not the arXiv one.

```
David R. Mortensen, Jordan Picone, Xinjian Li, and Kathleen Siminyu. Tusom2021: A Phonetically Transcribed Speech Dataset from an Endangered Language for Universal Phone Recognition Experiments. In Proc. Interspeech 2021, pages 3660-3664. doi:10.21437/Interspeech.2021-1435
```

## How the labels were made

### In short

People wrote these labels. The paper names no G2P step and no forced alignment, and East Tusom has no writing system to convert from, page 3660. One person with graduate training in phonetics and field linguistics wrote them first, over several passes. A second person checked all of them. The paper gives no measure of agreement. These are the paper's words, from page 3662.

```
The first author, who had graduate training in phonetics and field linguistics, elicited an item from the speaker, then attempted to imitate it until the speaker confirmed that it was correctly produced, then transcribed it.
```

```
The transcriptions and the recordings were subsequently aligned manually, using Praat TextGrids [9]. The first author then made a pass over the transcriptions. Subsequently, the first author made a partial second pass, correcting transcriptions of tone. The second author made a global pass, checking all transcriptions for correctness and consistency. Final consistency checks were done using a combination of automatic and manual methods (using Python scripts to extract all tokens transcribed with a particular phone, then using auditory inspection and acoustic analysis—using Praat—to winnow out distinctions that had been inconsistently noted in the previous pass, such as voicing in plosives).
```

```
We estimate that the production of the transcriptions alone, leaving aside elicitation, recording, and segmentation, took over 200 hours.
```

### Human or machine

Human, as far as the paper says. The only machine part it names is a set of scripts, on page 3662. They pulled out every token with a given phone, so that a person could check those tokens by ear and in Praat.

### Training of the people involved

* The first author had graduate training in phonetics and field linguistics, page 3662.
* The second author is listed under the Department of Linguistics, University of Pittsburgh, page 3660. The paper does not state this person's training.

### Number of annotators and agreement

The paper names two people. One wrote and corrected the labels. The other checked all of them. It does not say who did the final checks. It reports no agreement figure. So this dataset cannot measure agreement between annotators, which issue #10 of this repo asks about.

### Words first written at elicitation

The first transcription of a word was made before the recording, page 3662. Later passes corrected it. The paper does not say how many labels each pass changed. Tokens of one type can differ, as the paper says on page 3662.

```
In some cases, tokens of a type are not phonetically identical (and are not identically transcribed).
```

In the data, 721 groups of two or more tokens share a stem and a gloss. 34 of them have more than one `tone_dias` form. In 11 of the 34, each form has three or more tokens, so those look like different words with one gloss. Most of the others look like real differences between tokens, or errors, such as `verb17-19.wav` below.

## Transcription conventions

* Phones, not phonemes, page 3660. The paper does not use the words broad or narrow.
* IPA, as Unicode text. The joined sounds `t͡s` and `t͡ɕ` have a tie bar, 335 in all. Aspiration is the letter `ʰ`, and a nasal vowel is a vowel with `U+0303` after it.
* Two vowels in a row have no mark between them.
* Tones, three in all, high, low and falling, page 3660. `tone_dias` writes them as the combining marks `U+0301`, `U+0300` and `U+0302`. `tone_letters` writes them as Chao letters after each syllable, such as `˥˧`.
* No stress marks and no length marks.
* 340 labels have a space between words.

## Cleaning done by the authors

The final checks on page 3662 removed distinctions that had been marked unevenly, such as voicing in plosives. See the quote above. The paper also says that an uncorrected subset of this data was used in an earlier paper, page 3661.

## Cleaning done in this loader

* 137 WAV files have no label, so they are skipped and counted, in `self.skipped`.
* White space comes out of the IPA. This changes 343 labels, and the count of each split is in `self.cleaned`. 340 have spaces between words. 6 have the control character `U+001D`, and 3 of those also have a space. The repo's training code and `preprocess_ipa` also remove spaces, and PER would count them. The exact label is still in `raw_label`.
* No Unicode normalization. It would turn `a` followed by `U+0303` into the single letter `ã`, which `preprocess_ipa` drops whole.
* No phone is changed. No audio is trimmed or made louder. Audio is read with `soundfile` and made 16 kHz with the repo's `audio_resample`.

## Cleaning still needed

Each of these is left in, so that nobody's research labels change quietly.

* `verb17-19.wav` has the label `ɑ` in all three forms. The tokens next to it, with the same gloss, high, have `kət͡ɕʰuə`. It looks like a typing error. It is in train.
* `bp9-85.wav` is 0.223 s long and almost silent, with a peak of 0.0064 of full scale, for a label of 7 sounds. It is in train.
* 7 labels of `no_tones` still have tone on a vowel, 10 accented vowels in all. They are `bp12-6` to `bp12-9`, with `á`, and `bp2-70` to `bp2-72`, with `á` and `ỳ`. `preprocess_ipa` drops these vowels whole. They also put two bare tone marks, `U+0300` and `U+0301`, in `vocab` for the form with no tones.
* `g` is the plain letter in the 3 labels `animals1-22` to `animals1-24`, instead of the IPA letter `U+0261`. `preprocess_ipa` accepts it.
* `ʃ` appears in the 3 labels `verb13-38` to `verb13-40`, where the rest of the data has `ɕ`. `ɨ` appears 16 times next to the common `ɯ`. Whether these are real differences is not verified.
* `bp10-84.wav`, which has a label, is byte for byte the same file as `bp10-74.wav`, which has none. No two labeled items share audio.
* The 137 unlabeled files could be labeled, but only by listening. Why they were left out is not stated. All 137 are in `data/wav/transcriptions.tsv`, which looks like an earlier draft, with 774 rows. It writes `ʃ` and `ɨ` where `mapping.yml` writes `ɕ` and `ɯ`. `mapping.yml` has 637 of its rows. With those letters and a few marks folded together, 318 rows are equal and 124 are a different word. For 116 of those 124, the same word is in `mapping.yml` within 10 file numbers, in 8 stems. So the draft's rows are shifted in those stems, and it cannot be trusted file by file. 66 of the 137 have a number one more than a multiple of 4, which would fit the first of four tokens. Their pitch, by a rough estimate, is like that of the labeled clips. Its median is near 210 Hz. So they are probably the same speaker. The loader does not read this file.

## Splits

* `partition_data.py`, run on a copy with the pinned Python and packages, rebuilds `train.yml`, `dev.yml` and `test.yml` byte for byte. It groups tokens by their `tone_dias` label and shuffles the groups with seed 256. The first 70% of the groups are train, the next 10% dev, and the rest test.
* So no `tone_dias` label is in two splits. 10 `no_tones` labels are, since words that differ only in tone can land in different splits. They are `kəka`, `kətxuə`, `kətʰuə`, `kəxui`, `kʰəmɯə`, `kʰənt͡sɯ`, `kʰəŋuə`, `t͡ɕuŋkʰəma`, `ʔikətuəpi` and `ʔɯkʰr̥u`. With `tones="none"`, a few test labels also occur in train.
* The split is by label, not by speaker or session. 37 stems are in both train and test. For a model that never saw this language, this does not matter. For fine tuning it does.

## Where the paper and the data differ

* The paper's Section 3 says 2,393 tokens on page 3662. The abstract and Table 5 say 2,255, which is the number of labels. The disk has 2,392 WAV files. `data/wav` has 2,393 files if `transcriptions.tsv` is counted, which may be where that number comes from. Not verified.
* The paper says the recordings are at 22.05 kHz, page 3662. 219 files are at 16 kHz, all of the stems `bp4`, `bp5` and `bp6`.
* Table 5 gives 15,132, 2,133 and 4,283 phones. `partition_data.py` prints 15,136, 2,133 and 4,289 today. The differences, 4 and 6, are exactly the precomposed vowels `á` and `ỳ` in the `tone_dias` labels of train and test. panphon 0.22.2 counts each as two units. Those counts also count every space as one unit, 333, 38 and 76 of them. One guess is that an older panphon counted each as one unit. Not verified.
* Table 4 on page 3661 lists 10 vowel pairs. All 10 are in the data, as two vowel letters in a row in one word. The data also has `ui`, 43 times, and `ɨə` and `ae`, 3 times each.

## Symbols the evaluation drops

`evaluate(ipa, ipa)` gives PER 0 and FER 0 on every item of every form, and it never raises.

| Form | Code points | Lost in `preprocess_ipa` | Share | Labels affected | PER of a perfect prediction with no tones |
|---|---|---|---|---|---|
| `no_tones` | 16,949 | 10 | 0.06% | 7 | 0 |
| `tone_dias` | 23,341 | 6,405 | 27.44% | 2,241 | 0.2738 |
| `tone_letters` | 29,733 | 12,835 | 43.17% | 2,241 | 0.4275 |

* The loss in `no_tones` is the 10 accented vowels listed under Cleaning still needed. `preprocess_ipa` drops a precomposed vowel whole.
* Every tone mark is lost in the two tone forms. So FER cannot see tone today, and PER punishes a model for every tone mark it does not write.
* In `tone_letters`, `preprocess_ipa` also loses `U+0303`, which marks a nasal vowel, 38 times, and the syllable mark `U+0329` 3 times. In `tone_dias`, it loses `U+0303` 3 times.
* `simplify_ipa` leaves tone marks in 2,241 labels of `tone_dias` and 2,173 of `tone_letters`, although it calls `remove_tones_and_stress`.
* FER divides by the code points of the cleaned label, not by phones. The cleaned `no_tones` labels have 1.166 code points per panphon unit.

## Resampling

`audio_resample` has no low pass filter, so sound above 8 kHz folds back at 16 kHz. We measured the 2,173 files at 22,050 Hz. The share of their energy above 8 kHz has a median of 0.00053, about minus 33 dB. Its 95th percentile is 0.015, and its largest value is 0.227.

## What it is good for

* A test of whether a model hears the phones of a language it has never seen. A person trained in phonetics wrote the labels, and a second person checked them.
* Tusom2021 is not listed in the training data of `KoelLabs/xlsr-english-01`, and its base model's card names Common Voice with no list of languages. We read those cards on Nov 7. The pages could not be reached on Nov 9, so this was not checked again. Whether Common Voice has this language is not verified.
* A small tuning set. The paper's best PER after tuning is 33%, from a baseline of 72%, page 3663. Its zero shot results are on page 3662.

```
test        eng         88.4%    9.6%     14.6%     64.2%
            lg_ind      71.9%    2.5%     24.0%     45.4%
all         eng         88.9%    9.8%     14.7%     64.3%
            lg_ind      69.2%    3.2%     23.7%     42.4%
```

Those rows are from Table 6, with the columns PER, Add, Del and Sub. They use `no_tones` with rare phones replaced by similar common ones, page 3662, so they are not scores on these raw labels.

## What the data does not cover

* Other speakers, men, other ages.
* Running speech. A few items are short sentences, and 75 labels have 3 or more words.
* Any other recording setup.
* Phone times. The paper says the words were aligned by hand in Praat, page 3662, and the release has no TextGrid file.

## How we checked

Scripts are kept outside this PR. They read every label and every WAV file. The loader's own tests ran on every item of every split and form, and compared each item with `mapping.yml` and with its WAV file. Results are the same under two hash seeds and between commits. Nobody has listened to the clips against their labels yet, so we do not claim that the audio matches the labels.

## Sources

* The data repo, https://github.com/dmort27/tusom2021, commit `e168dfb90f2b03d433742d5ba5ef5e5731686610`, read Nov 9, 2026.
* The paper as a PDF, https://www.isca-archive.org/interspeech_2021/mortensen21_interspeech.pdf, read Nov 9, 2026.
* Glottolog, https://glottolog.org/resource/reference/id/636054, read Nov 9, 2026.
* The project page, https://llab-cmu.github.io/projects/tusom-project.html, read Nov 4, 2026.
* The model cards, https://huggingface.co/KoelLabs/xlsr-english-01 and https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft, read through a web reader on Nov 7, 2026.
