from transformers import pipeline

from phonemizer.backend.espeak.wrapper import EspeakWrapper

_ESPEAK_LIBRARY = "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib"
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

# load model and processor
pipe = pipeline(
    "automatic-speech-recognition",
    # model="facebook/wav2vec2-lv-60-espeak-cv-ft",
    # model="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    model="ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",
)

print(pipe("./alexIsConfused.wav"))
