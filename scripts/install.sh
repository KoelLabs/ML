# Tested with Python 3.8.10 on MacOS 14.4.1

# dependencies
if [ "$(uname)" == "Darwin" ]; then
    # Do something under Mac OS X platform  
    brew install espeak
    brew install ffmpeg      
    brew install portaudio
    pip install pyaudio
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
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

# install https://github.com/jhasegaw/phonecodes
cd ./repos
git clone https://github.com/jhasegaw/phonecodes.git
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
