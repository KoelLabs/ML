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
