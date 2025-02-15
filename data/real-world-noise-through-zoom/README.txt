# Real-world noise settings
- 5 different real world noise settings (bedroom, crowded room, background music, rain, road with cars)
- 2 different speakers
- various microphone distances (6 inches, 24 inches)
- 32 total samples with different phrases
- recorded through Zoom to simulate real-world linguistic fieldwork scenarios

### Processing audio files
- Recorded in m4a through Zoom
- Then converted to mono 16kHz wav files using ffmpeg:
    - for i in *.m4a; do ffmpeg -i "$i" -acodec pcm_s16le -ac 1 -ar 16000 "${i%.*}.wav"; done
    - rm *.m4a
- Then transcribed using
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" KoelLabs/xlsr-timit-c0 > "${i%.*}.c0.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" KoelLabs/xlsr-timit-c1 > "${i%.*}.c1.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" "ginic/gender_split_70_female_4_wav2vec2-large-xlsr-53-buckeye-ipa" > "${i%.*}.taguchi.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" "mrrubino/wav2vec2-large-xlsr-53-l2-arctic-phoneme" > "${i%.*}.rubino.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" "facebook/wav2vec2-lv-60-espeak-cv-ft" > "${i%.*}.facebook60.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/ipa_transcription/xlsr.py "$i" "vitouphy/wav2vec2-xls-r-300m-timit-phoneme" > "${i%.*}.vitou.txt"; done
    - for i in data/real-world-noise-through-zoom/*.wav; do python ./scripts/asr/azure_speech.py "$i" > "${i%.*}.words.txt"; done
        - Manual verification
