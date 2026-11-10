# Indic TIMIT

These notes are from Nov 9, 2026. Every number comes from the 18 archives of the SPIRE Lab in `.data/IndicTIMIT`, unless a line says otherwise. Facts from the 2019 paper give its section, since its PDF text has no page numbers. The papers were read through a web reader, which cuts long sentences, so a quote may stop mid sentence. Each was checked twice. What we could not check says not verified.

## The labels, in short

* This loader is for human phone labels. The 2019 paper says two linguists wrote them by hand for 2,342 recordings, one of each sentence, and that the data comes with them. The SPIRE README also says linguists labeled a subset.
* They are not in the 18 archives as we downloaded them. Each archive holds only WAV files, their text and a list of the sentences.
* So until the label files are found, the loader stops at setup with an error that says this. It never puts machine made labels in their place.
* When they are found, one function, `read_phone_labels`, gets a parser, and the loader returns only the recordings that have a label, at most 2,342 of the 187,360. The others are left out by design.

## Words used below

* A nativity is a native language of the speakers. Each has one archive, named with a code such as `BEN`.
* A sentence is one of the 2,342 distinct TIMIT sentences. Every speaker read all of them.
* A recording is one speaker reading one sentence, in one WAV file.

## Loader

`scripts/data_loaders/IndicTIMIT.py`, class `IndicTIMITDataset`.

* `split` is `all` or a nativity code, such as `BEN`. No speaker is in two nativities. The release has no train or test split, and the 2019 paper's, from 63 speakers it does not name, cannot be rebuilt. The paper counts the labeled recordings by group, not by nativity, so a nativity split may get few or none, and one with none stops with an error.
* The human labels are read from `.data/IndicTIMIT/phone_labels`, a name of ours. Setup stops if the folder is missing or empty, and also if it holds files, since the loader does not guess a format we have not seen.
* `read_phone_labels` must give every label it finds, and the loader keeps those of the split.
* `self.recordings` lists every recording. `self.items` holds the ones with a human label, and `self.skipped` counts the rest.
* The IPA will be the human label without its spaces and its word marks, `∼`. The label as given is `raw_label`.
* `include_timestamps=True` fails, since the data has no phone times.
* `include_speaker_info=True` gives `id`, `speaker`, `sex`, `nativity`, `native_language`, `region`, `group`, `sentence`, `raw_label` and `source`.
* `include_text=True` gives the sentence, as `Transcripts_<CODE>` writes it, without its last space.
* Setup stops with an error that says what to do when a folder, a transcript or a WAV file is missing, a transcript is cut inside a line, or a line is broken or repeated. It also stops on a label that does not fit its recordings or is not IPA. A WAV file with no line is skipped and counted.

## Report

| Item | Value | Source |
|---|---|---|
| Items today | none, setup stops with no human label on disk | Loader |
| Items with the labels | up to 2,342, one per sentence, 512, 512, 295, 511 and 512 from groups 1 to 5 | Paper, section III.A |
| Clip seconds, all recordings | shortest 1.194, median 4.267, longest 15.786 | Data |
| Setup time | 1.12 s for `BEN`, 4.83 s for `all`, best of 3 | Loader with made up labels, on a Linux VM, Nov 9 |
| Peak memory | 506 MB for `all` | Same |

The table below is from the files. Its hours, rounded, equal the README's.

| Code | Native language | Group | Speakers | Recordings | Hours | WAV format |
|---|---|---|---|---|---|---|
| `ASS` | Assamese | 1 | 1, 1 F, 0 M | 2,342 | 3.02 | 44.1 kHz, mono, 16 bit |
| `BEN` | Bengali | 1 | 8, 4 F, 4 M | 18,736 | 23.52 | 44.1 kHz, mono, 16 bit |
| `DIM` | Dimasa | 1 | 1, 1 F, 0 M | 2,342 | 2.93 | 44.1 kHz, mono, 16 bit |
| `GUJ` | Gujarati | 3 | 2, 0 F, 2 M | 4,684 | 6.90 | 44.1 kHz, mono, 16 bit |
| `HIN` | Hindi | 2 | 13, 6 F, 7 M | 30,446 | 35.30 | 44.1 kHz, mono, 16 bit |
| `KAN` | Kannada | 4 | 8, 5 F, 3 M | 18,736 | 24.54 | 44.1 kHz, mono, 16 bit |
| `KON` | Konkani | 3 | 2, 2 F, 0 M | 4,684 | 6.12 | 44.1 kHz, mono, 16 bit |
| `MAI` | Maithili | 1 | 1, 0 F, 1 M | 2,342 | 2.74 | 44.1 kHz, mono, 16 bit |
| `MAL` | Malwi | 2 | 1, 0 F, 1 M | 2,342 | 3.32 | 44.1 kHz, mono, 16 bit |
| `MAR` | Marathi | 3 | 10, 5 F, 5 M | 23,420 | 28.02 | 44.1 kHz, mono, 16 bit |
| `MLY` | Malayalam | 5 | 8, 5 F, 3 M | 18,736 | 24.40 | 44.1 kHz, mono, 16 bit |
| `MOG` | Mog | 1 | 1, 0 F, 1 M | 2,342 | 3.10 | 44.1 kHz, mono, 16 bit |
| `NEP` | Nepali | 1 | 1, 1 F, 0 M | 2,342 | 2.96 | 44.1 kHz, mono, 16 bit |
| `ODI` | Odia | 1 | 2, 1 F, 1 M | 4,684 | 5.24 | 44.1 kHz, mono, 16 bit |
| `PUN` | Punjabi | 2 | 2, 2 F, 0 M | 4,684 | 5.59 | 44.1 kHz, mono, 16 bit |
| `SAU` | Saurashtra | 3 | 2, 1 F, 1 M | 4,684 | 6.16 | 44.1 kHz, mono, 16 bit |
| `TAM` | Tamil | 5 | 9, 4 F, 5 M | 21,078 | 26.23 | 44.1 kHz, mono, 16 bit |
| `TEL` | Telugu | 4 | 8, 3 F, 5 M | 18,736 | 24.37 | 44.1 kHz, mono, 16 bit |
| all | | | 80, 41 F, 39 M | 187,360 | 234.47 | 44.1 kHz, mono, 16 bit |

## Speakers

Section II.A of the 2019 paper describes the speakers in these words.

```
we consider a total of 16 subjects from each of these groups with equal male to female ratio
The age of the subjects vary from 18 to 60 years with an average age of 25.42 years with standard deviation of 6.05 years.
The subjects are mostly undergraduate or post graduate students
```

The sentence goes on to name the Indian Institute of Science in Bangalore. The paper says the speakers gave consent and got a token payment. The release has no speaker file and no age. A folder name such as `F30` is the speaker ID. Its first letter, F or M, is given as `sex`, since in the 9 rows of Table I that match, below, its counts fit those of women and men. Names are left out of the loader, these notes and all outputs.

Table I of the paper counts speakers by native language. The release has 80 too, but 41 F and 39 M, and 15 to 17 in a group.

* Groups 2, 3 and 4 match, 16 speakers each, 8 F and 8 M. So do their rows if Table I counts Malwi with Hindi, 14 with 8 M, and Saurashtra with Gujarati, 4 with 3 M.
* Group 5 has 17, 9 F and 8 M. `TAM` has 9, 5 M and 4 F, where Table I has 8, 5 M and 3 F.
* Group 1 has 15, 8 F and 7 M, where Table I has 16, with 8 M. Table I has a Manipuri man, and the release has no Manipuri archive. It gives Assamese 2 and Oriya 3, where `ASS` has 1 and `ODI` has 2, and no row for Dimasa or Mog.
* The other 9 rows match in speakers and sex.

## Recording

Section II.B describes the recording in these words.

```
We consider Zoom H6 mixer along with Rode procaster microphone in the recording.
We recorded all the stimuli in a quiet room.
All the recordings are done in 48000Hz sampling rate with 16bit PCM format.
The audio of each sentence is downsampled to 16000Hz
the operator carefully listen to the subject's speech to spot error if there is any.
If any error occurs, the subject is asked to read the sentence once again.
```

The sentences were read in 16 parts, 15 of 150 and one of 92. `transcript_details.txt` names each sentence, such as `Phoneme_L5_0047.WRD`, in 16 groups called Phoneme, Stress and Intonation that fit those parts. The names are not explained, and no `.WRD` file is in the release.

## License and how to get the data

The data is requested through a form on https://spiredatasets.ee.iisc.ac.in/indictimitcorpus. Its README, saved on Nov 9, names no license or terms, and we could not read the form. So the terms, and whether data made from it may be shared, are not known. The loader's header says how to unpack the archives.

## Citation

The README on the SPIRE page asks for this citation.

```
C. Yarra, R. Aggarwal, A. Rajpal, and P. K. Ghosh, "Indic TIMIT and Indic English lexicon: A speech database of Indian speakers using TIMIT stimuli and a lexicon from their mispronunciations," 2019 22nd Conference of the Oriental COCOSDA International Committee for the Co-ordination and Standardisation of Speech Databases and Assessment Techniques (O-COCOSDA), Cebu, Philippines, 2019, pp. 1-6, doi: 10.1109/O-COCOSDA46868.2019.9041230.
```

## How the human labels were made

### Human or machine

Human. Two linguists wrote them by hand in a web tool that played each recording. These are the paper's words, from the abstract and section III.A.

```
Further, the corpus also contains phoneme transcriptions for a sub-set of recordings,
which are manually annotated by two linguists reflecting speaker’s pronunciation.
The stimuli for annotation are selected such that it covers the entire set of 2342 unique sentences
in the TIMIT maintaining one recording per sentence.
Out of 2342 selected stimuli, a total of 512, 512, 295, 511 and 512 stimuli are considered randomly
```

The five counts are for the five groups, in order. Which speaker each recording comes from is not given. Section III.A.1 says the tool had Play, Previous, Next and Submit buttons, and refused a label that broke the instructions.

### What the linguists wrote

They typed IPA in the tool, with `∼` between words and none between words they heard as one. The slides' one example also puts a space between phones. The instructions, from section III.A and slide 13, are these.

```
Use ‘∼’ to indicate the word boundaries in the phoneme transcriptions.
In case of co-articulation between the words, merge those co-articulated words with ‘-’ symbol
in the modified text box (Don’t split the words).
```

The `-` goes in a box for the text, not in the phones. The example comes out of the PDF as plain letters, such as `I` and `tS`. They may be IPA symbols in an old font, so the characters of the real files are not known. The 2023 paper says its later labels use 108 IPA symbols, and the 2019 paper gives no number.

### Where the files are said to be

Section III.A.2 says the data comes with the labels in IPA, in ARPAbet and in a set of 40 phones, with tables between them, in these words.

```
In Indic TIMIT, we provide files containing original phoneme transcriptions, as well as in ARPAbet format
and in ARPAbet mapped with 40 phoneme set.
```

None of these files is in the 18 archives, and the paper gives no file names. Nor is the Indic English lexicon, which is said to come with the data too.

### Training of the people involved

Section III.A gives their degrees and work places, in these words.

```
One annotator has a PhD degree in linguistics and is working in central institute of Indian languages (CIIL), Mysore, India.
The other annotator has a MSc degree in linguistics and is working in CIIL, Mysore, India.
```

We found no word in the paper on training for this task, or on how the two linguists split the work.

### Agreement

The 2019 paper reports no agreement between the two linguists. Its Table II, 51.77 percent of phones the same as TIMIT's own, is not agreement. The SPECOM 2023 paper says one of them wrote 13,632 more labels, 15,974 in all, and labeled 200 files per group twice, with a mean Cohen's Kappa of 0.827 between the passes. It does not say where those labels are.

## Cleaning done and still needed

Section II.B says an operator listened as each sentence was read, and asked for it again after an error.

The loader does no cleaning. It trims no audio and drops no recording for how it sounds. Our checks found no problem in the 187,360 WAV files.

* Each has a plain 44 byte header with the right size, and gives the same samples with `soundfile` as with Python's own `wave` module.
* None is silent, and no two hold the same audio.
* Each transcript line has its WAV file and each WAV file its line. The 2,342 sentences are written the same in all 18 transcripts.

Still needed.

* The human labels, and maybe the 13,632 of the 2023 paper. Then a check of their symbols, of their count in each group against the paper's, and of each label against its recording.
* Listening. Nobody has listened to a recording yet.
* 16 recordings are very quiet. Their loudest 20 ms is more than 40 dB below full scale, and 60.45 dB below in `MLY` `F27` sentence 26. Seven of them are from `MAR` `F22`. They may hold little or no speech.
* 1,726 recordings reach full scale at least once, 671 of them 10 times or more, and 1,260 come from 8 speakers, such as `HIN` `M33`. They may be clipped.

## Where the papers and the data differ

* The paper says the audio was recorded at 48,000 Hz and downsampled to 16,000 Hz. Every file in the release is 44,100 Hz, and those checked hold sound above 8 kHz, which a 16 kHz file cannot. The 187,360 files have only 254 lengths. All but 4 last a whole number of blocks of 4,096 samples at 48,000 Hz, to within 45 samples at 44,100 Hz. That fits a recording at 48,000 Hz turned into 44,100 Hz.
* The paper and the README give about 240 hours, and the files hold 234.47.
* The paper names its files `LANG GEN-ID AGE STIMULIno.wav`. The release names them `IISc_IndicTIMIT_<ID>_<n>.wav`, with no age.
* The README lists `M28` under both `HIN` and `MAI`, and leaves out `F31`, which is in the `HIN` archive. Its hours and counts match the files.
* The README gives MOG as the language of `MOG`, and the 2023 paper Mog, which the loader uses.

## Resampling

The loader resamples to 16,000 Hz with `audio_resample` in `scripts/core/audio.py`, as `VoxAngeles.py` does. It draws straight lines between samples with no low pass filter, so sound above 8 kHz folds back below it. In every 50th sentence of every speaker, 3,760 files, that sound is a median share of 0.000921 of the energy, 30.4 dB down, and at most 0.086738. `audio_bytes_to_array` in the same file gives the same samples on all 3,760.

## What it is good for

* With the human labels, a test of phone recognition on Indian English, up to 2,342 recordings, one per sentence, from all five groups.
* All speakers read the same sentences, so they can be compared word for word.

## What the data does not cover

* More than one label for each sentence, phone times, or running speech.
* Speakers from outside one institute, or facts about them beyond the paper's.

## How we checked

Scripts are kept outside this PR. With no human label on disk, the tests gave every recording a made up label through `read_phone_labels`. Then they read all 187,360 items, audio included, and checked each against its transcript line and WAV file, under two hash seeds. The tests ran on each commit's code before it was made, and every item stayed the same. A few made up labels tested the filter, and broken copies of the data tested each error.

## Sources

* The data, the 18 archives `IISc_IndicTIMIT_<CODE>.tar.gz` from the SPIRE Lab, unpacked on Nov 9, 2026, and the README of https://spiredatasets.ee.iisc.ac.in/indictimitcorpus, saved that day.
* The 2019 paper, https://spire.ee.iisc.ac.in/assets/PDFs/papers_pdf/Chiranjeevi_COCOSDA_2019.pdf, and its slides, https://spire.ee.iisc.ac.in/assets/PDFs/slides/Chiranjeevi_COCOSDA_2019_slides.pdf, read Nov 9, 2026.
* The SPECOM 2023 paper, https://spire.ee.iisc.ac.in/assets/PDFs/papers_pdf/Study_of_Indian_English_Pronunciation_variabilities_relative_to_Received_Pronunciation.pdf, read Nov 9, 2026. arXiv 2204.06502, with the same title, was not read.
