# Tested with Python 3.8.10 on MacOS 14.4.1

# create a virtual environment
python -m venv ./venv

# activate the virtual environment
source ./venv/bin/activate

# dependencies
pip install sherpa-ncnn
pip install sounddevice

# install https://huggingface.co/bookbot/pruned-transducer-stateless7-streaming-id
mkdir -p sherpa-models
cd sherpa-models
git lfs install
git clone https://huggingface.co/bookbot/sherpa-ncnn-pruned-transducer-stateless7-streaming-id.git
wget https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2
tar xvf sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2
