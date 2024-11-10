#!/usr/bin/env python3

from transformers import pipeline
import sys
import os
from tempfile import NamedTemporaryFile


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.eval_tests.panphon_model_eval import panphon_model_eval  # Import the function
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from scripts.audio import audio_record_to_file

# list of English IPA XLSR models that work
MODEL_IDS = [
    "facebook/wav2vec2-lv-60-espeak-cv-ft",  # Samir's recommended best for English
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft",  # very similar to 60
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns",  # Recommended by Samir, but not for English
    "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa-plus-2000",  # better quality, slightly slower but also not for English
    "ginic/wav2vec-large-xlsr-en-ipa",  # OK quality, also annotates stressed syllables with ' as the only (ginic) model to do so
    "ginic/data_seed_4_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/gender_split_30_female_5_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/gender_split_70_female_5_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/vary_individuals_old_only_3_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/vary_individuals_young_only_3_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "ginic/hyperparam_tuning_1_wav2vec2-large-xlsr-buckeye-ipa",  # ginic models are all very similar
    "speech31/wav2vec2-large-TIMIT-IPA",  # works quite well and has word boundaries as the only model
    "speech31/wav2vec2-large-TIMIT-IPA2",  # works quite well but no word boundaries
    "speech31/wav2vec2-large-english-TIMIT-phoneme_v3",  # works quite well but no word boundaries
    "speech31/XLS-R-300m-english-ipa",  # slightly weirder spelling
    "speech31/wavlm-large-english-ipa",  # adds extra sounds that are not there
    "speech31/hubert-base-english-ipa",  # adds extra sounds
    "snu-nia-12/wav2vec2-large_nia12_phone-ipa_english",  # works quite well
    "Jubliano/wav2vec2-large-xls-r-300m-ipa",  # quite big and slow to load, very weird transcriptions
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-nl",  # smaller, still weird
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-INTERNATIONAL1.5",  # OK, a bit unconventional spelling
    "Jubliano/wav2vec2-large-xls-r-300m-ipa-INTERNATIONAL1.9.2WithoutSpaces",  # not bad, not good
    "vitouphy/wav2vec2-xls-r-300m-timit-phoneme" # specifically for arpabet phoneme prediction, works well, similar to snu-nia-12
]


pipelines = {}

def xlsr_transcribe(input_path, model_id=MODEL_IDS[0]):
    pipelines[model_id] = pipelines.get(model_id) or pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device="cpu",
    )
    return pipelines[model_id](input_path).get("text", "")

def evaluate_xlsr(input_path, ground_truth, model_id=MODEL_IDS[0]):
    # Transcribe the input audio file using the selected model
    transcription = xlsr_transcribe(input_path, model_id)
    print(f"Transcription from model '{model_id}': {transcription}")
    # remove the spaces between the phonemes
    transcription = transcription.replace(" ", "")
    predicted = transcription
    label = ground_truth

    # Call panphon_model_eval with label and predictedipa
    results = panphon_model_eval(label, predicted)

    # Output results
    print("Evaluation Results:")
    print(f"DTW distance: {results['dtw_distance']}")
    print(f"CER: {results['cer_score']}")
    
    return results
    
    

def main(args):
    # Get model_id from command line argument or use default model if none provided
    model_id = args[0] if len(args) > 0 else MODEL_IDS[0]

    # Default input path and ground truth if not provided
    input_path = args[1] if len(args) > 1 else "/home/arunasrivastava/ML/data/TIMIT_sample_0.wav"
    ground_truth = args[2] if len(args) > 2 else "ðɨaɪɹeɪtʔækɚstɑmpəweɪʔɨɾiɑɾɨkli"
    print("Ground truth:  ð ɨ a ɪ ɹ e ɪ t ʔ æ k ɚ s t ɑ m p ə w e ɪ ʔ ɨ ɾ i ɑ ɾ ɨ k l i")
    print(f"Starting evaluation on TIMIT audio sample: {input_path}")
    
    # Evaluate using the specified model, input path, and ground truth
    evaluate_xlsr(input_path, ground_truth, model_id)

if __name__ == "__main__":
    main(sys.argv[1:])
