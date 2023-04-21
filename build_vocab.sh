mkdir -p data
if [ ! -e data/text8 ]; then
    wget http://mattmahoney.net/dc/text8.zip -O data/text8.zip
  cd data
  unzip text8.zip
  rm text8.zip
  cd ..
fi

python pw2v/vocab.py --file data/text8 --save data/text8.vocab