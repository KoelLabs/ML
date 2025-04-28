# Tested with Python 3.8.10 on MacOS 14.4.1

# dependencies
if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific  
    brew install espeak
    brew install ffmpeg      
    brew install portaudio
    pip install pyaudio
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # GNU/Linux specific
    sudo apt-get update
    sudo apt-get install ffmpeg espeak-ng libportaudio2 python3-pyaudio
else
    echo "please install espeak and ffmpeg manually for this OS"
fi

pip install -r requirements.txt 


# create gitignored directories
mkdir -p ./.data
mkdir -p ./models
mkdir -p ./repos

# install https://huggingface.co/bookbot/pruned-transducer-stateless7-streaming-id
cd ./models
mkdir -p sherpa-models
cd sherpa-models
git lfs install
git clone https://huggingface.co/bookbot/sherpa-ncnn-pruned-transducer-stateless7-streaming-id.git
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2
tar xvf sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2
rm sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2
cd ../../

# install https://github.com/PKlumpp/phd_model
cd ./repos
git clone https://github.com/PKlumpp/phd_model.git
cd ../

# install https://github.com/xenova/transformers.js
cd ./repos
git clone https://github.com/xenova/transformers.js.git
cd ../

# install ./.data/TIMIT.zip from https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech?resource=download
curl -L -o ./.data/TIMIT.zip\
  https://www.kaggle.com/api/v1/datasets/download/mfekadu/darpa-timit-acousticphonetic-continuous-speech

# download https://affective-meld.github.io/
cd ./.data
wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
cd ../

# download https://www.kaggle.com/rtatman/speech-accent-archive
curl -L -o ./.data/speech-accent-snake-snack.zip\
  https://www.kaggle.com/api/v1/datasets/download/rtatman/speech-accent-archive

# download Kokoro-Speech-Dataset
curl -L -o ./.data/Kokoro-Speech-Dataset.zip\
  https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/kokoro-speech-v1_3.zip
mkdir -p ./.data/Kokoro-Speech-Dataset
curl -L -o ./.data/Kokoro-Speech-Dataset/meian-by-soseki-natsume.zip https://www.archive.org/download/meian_1403_librivox/meian_1403_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/kokoro-by-soseki-natsume.zip http://www.archive.org/download//kokoro_natsume_um_librivox/kokoro_natsume_um_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/inakakyoshi-by-katai-tayama.zip http://www.archive.org/download//inakakyoshi_1311_librivox/inakakyoshi_1311_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/nowaki-by-soseki-natsume.zip http://www.archive.org/download/nowaki_um_librivox/nowaki_um_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/kusamakura-by-soseki-natsume.zip http://www.archive.org/download//kusamakura_1311_librivox/kusamakura_1311_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/botchan-by-soseki-natsume-2.zip http://www.archive.org/download//botchan_1310_librivox/botchan_1310_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/gan-by-ogai-mori.zip http://www.archive.org/download//gan_1311_librivox/gan_1311_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/umareizuru-nayami-by-takeo-arishima.zip http://www.archive.org/download/umareizuru_nayami_ez_1302/umareizuru_nayami_ez_1302_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/garasudono-uchi-by-natsume-soseki.zip http://www.archive.org/download/garasudonouchi_1208_librivox/garasudonouchi_1208_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/eijitsu-syohin-by-soseki-natsume.zip http://www.archive.org/download/eijitsu_syohin_um_librivox/eijitsu_syohin_um_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/futon-by-katai-tayama.zip http://www.archive.org/download/futon_ek_1303_librivox/futon_ek_1303_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/kouyahijiri-by-kyoka-izumi.zip http://www.archive.org/download/kouyahijiri_1303_librivox/kouyahijiri_1303_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/gongitsune-by-nankichi-niimi.zip http://archive.org/download/gongitsune_um_librivox/gongitsune_um_librivox_64kb_mp3.zip
curl -L -o ./.data/Kokoro-Speech-Dataset/caucasus-no-hagetaka-by-yoshio-toyoshima.zip http://www.archive.org/download/caucasus_no_hagetaka_um_librivox/caucasus_no_hagetaka_um_librivox_64kb_mp3.zip

# download CORAAL https://oraal.github.io/coraal
mkdir -p ./.data/CORAAL/phonemes
mkdir -p ./.data/CORAAL/audio
mkdir -p ./.data/CORAAL/transcripts
curl -L -o ./.data/CORAAL/phonemes/DCA.zip http://lingtools.uoregon.edu/coraal/aligned/DCA_MFA_2019.06.zip
curl -L -o ./.data/CORAAL/phonemes/DCB.zip http://lingtools.uoregon.edu/coraal/aligned/DCB_MFA_2019.06.zip
curl -L -o ./.data/CORAAL/phonemes/PRV.zip http://lingtools.uoregon.edu/coraal/aligned/PRV_MFA_2019.06.zip
curl -L -o ./.data/CORAAL/phonemes/ROC.zip http://lingtools.uoregon.edu/coraal/aligned/ROC_MFA_2019.06.zip

curl -L -o ./.data/CORAAL/ATL-metadata.txt http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_metadata_2020.05.txt
curl -L -o ./.data/CORAAL/audio/ATL_1.tar.gz http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part01_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ATL_2.tar.gz http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part02_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ATL_3.tar.gz http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part03_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ATL_4.tar.gz http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_audio_part04_2020.05.tar.gz
curl -L -o ./.data/CORAAL/transcripts/ATL.tar.gz http://lingtools.uoregon.edu/coraal/atl/2020.05/ATL_textfiles_2020.05.tar.gz

curl -L -o ./.data/CORAAL/DCA-metadata.txt http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_metadata_2018.10.06.txt
curl -L -o ./.data/CORAAL/audio/DCA_1.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part01_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_2.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part02_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_3.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part03_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_4.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part04_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_5.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part05_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_6.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part06_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_7.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part07_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_8.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part08_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_9.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part09_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCA_10.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_audio_part10_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/transcripts/DCA.tar.gz http://lingtools.uoregon.edu/coraal/dca/2018.10.06/DCA_textfiles_2018.10.06.tar.gz

curl -L -o ./.data/CORAAL/DCB-metadata.txt http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_metadata_2018.10.06.txt
curl -L -o ./.data/CORAAL/audio/DCB_1.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part01_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_2.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part02_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_3.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part03_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_4.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part04_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_5.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part05_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_6.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part06_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_7.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part07_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_8.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part08_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_9.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part09_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_10.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part10_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_11.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part11_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_12.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part12_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_13.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part13_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DCB_14.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_audio_part14_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/transcripts/DCB.tar.gz http://lingtools.uoregon.edu/coraal/dcb/2018.10.06/DCB_textfiles_2018.10.06.tar.gz

curl -L -o ./.data/CORAAL/DTA-metadata.txt http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_metadata_2023.06.txt
curl -L -o ./.data/CORAAL/audio/DTA_1.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part01_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_2.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part02_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_3.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part03_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_4.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part04_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_5.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part05_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_6.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part06_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_7.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part07_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_8.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part08_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_9.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part09_2023.06.tar.gz
curl -L -o ./.data/CORAAL/audio/DTA_10.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_audio_part10_2023.06.tar.gz
curl -L -o ./.data/CORAAL/transcripts/DTA.tar.gz http://lingtools.uoregon.edu/coraal/dta/2023.06/DTA_textfiles_2023.06.tar.gz

curl -L -o ./.data/CORAAL/LES-metadata.txt http://lingtools.uoregon.edu/coraal/les/2021.07/LES_metadata_2021.07.txt
curl -L -o ./.data/CORAAL/audio/LES_1.tar.gz http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part01_2021.07.tar.gz
curl -L -o ./.data/CORAAL/audio/LES_2.tar.gz http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part02_2021.07.tar.gz
curl -L -o ./.data/CORAAL/audio/LES_3.tar.gz http://lingtools.uoregon.edu/coraal/les/2021.07/LES_audio_part03_2021.07.tar.gz
curl -L -o ./.data/CORAAL/transcripts/LES.tar.gz http://lingtools.uoregon.edu/coraal/les/2021.07/LES_textfiles_2021.07.tar.gz

curl -L -o ./.data/CORAAL/PRV-metadata.txt http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_metadata_2018.10.06.txt
curl -L -o ./.data/CORAAL/audio/PRV_1.tar.gz http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part01_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/PRV_2.tar.gz http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part02_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/PRV_3.tar.gz http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part03_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/audio/PRV_4.tar.gz http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_audio_part04_2018.10.06.tar.gz
curl -L -o ./.data/CORAAL/transcripts/PRV.tar.gz http://lingtools.uoregon.edu/coraal/prv/2018.10.06/PRV_textfiles_2018.10.06.tar.gz

curl -L -o ./.data/CORAAL/ROC-metadata.txt http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_metadata_2020.05.txt
curl -L -o ./.data/CORAAL/audio/ROC_1.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part01_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ROC_2.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part02_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ROC_3.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part03_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ROC_4.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part04_2020.05.tar.gz
curl -L -o ./.data/CORAAL/audio/ROC_5.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_audio_part05_2020.05.tar.gz
curl -L -o ./.data/CORAAL/transcripts/ROC.tar.gz http://lingtools.uoregon.edu/coraal/roc/2020.05/ROC_textfiles_2020.05.tar.gz

curl -L -o ./.data/CORAAL/VLD-metadata.txt http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_metadata_2021.07.txt
curl -L -o ./.data/CORAAL/audio/VLD_1.tar.gz http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part01_2021.07.tar.gz
curl -L -o ./.data/CORAAL/audio/VLD_2.tar.gz http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part02_2021.07.tar.gz
curl -L -o ./.data/CORAAL/audio/VLD_3.tar.gz http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part03_2021.07.tar.gz
curl -L -o ./.data/CORAAL/audio/VLD_4.tar.gz http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_audio_part04_2021.07.tar.gz
curl -L -o ./.data/CORAAL/transcripts/VLD.tar.gz http://lingtools.uoregon.edu/coraal/vld/2021.07/VLD_textfiles_2021.07.tar.gz

# Download CMU ARCTIC dataset (individual speakers)
echo "Downloading CMU ARCTIC dataset..."
mkdir -p .data/CMU_ARCTIC && for spk in aew ahw aup awb axb bdl clb eey fem gka jmk ksp ljm lnh rms rxr slp slt; do wget -q --show-progress "http://festvox.org/cmu_arctic/packed/cmu_us_${spk}_arctic.tar.bz2" -O - | tar -xj -C .data/CMU_ARCTIC; done

# Download DoReCo Southern British English data
mkdir -p .data/DoReCo
curl -L https://sharedocs.huma-num.fr/?module=weblinks&section=public&multidownload=1&id=eGPAXNlp1L6E2aP9efkr4YCXiXFlDLI3 > ./.data/DoReCo/sout3282_audio_core_v2.zip
curl -L https://multicast.aspra.uni-bamberg.de/data/audio/english/wav/mc_english__wav.zip > ./.data/DoReCo/sout3282_audio_v2.zip
# curl -L https://sharedocs.huma-num.fr/wl/?id=JHfkc54sOLrj1zgkihIf9qKkrcfuQbsr&fmode=download > ./.data/DoReCo/sout3282_annotations_v2.zip
curl -L https://sharedocs.huma-num.fr/wl/?id=7XBW8mmwQkeZeYN18EnEiNpoeLFirrvi&fmode=download > ./.data/DoReCo/sout3282_annotations_v2.zip
