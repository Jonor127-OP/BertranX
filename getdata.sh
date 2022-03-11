#!/bin/bash
set -e

# Set absolute path
export PYTHONPATH="$PWD/dataset/:$PWD/dataset/data_conala/:$PWD/model/:$PWD/dataset/data_conala/conala-corpus/"
echo "$PYTHONPATH"

# Get the data
echo "download CoNaLa dataset"
wget http://www.phontron.com/download/conala-corpus-v1.1.zip
unzip conala-corpus-v1.1.zip -d ./dataset/data_conala
rm -r conala-corpus-v1.1.zip
echo "CoNaLa done"

#echo "download CodeSearchNet dataset"
#wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
#unzip python.zip -d ./dataset/data_github
#rm -r python.zip
#echo "CodeSearchNet done"

#echo "download APPS dataset"
#wget https://people.eecs.berkeley.edu/\~hendrycks/APPS.tar.gz --no-check-certificate
#tar -xzf APPS.tar.gz -C dataset
#rsync -a dataset/apps_dataset/APPS dataset/data_apps
#rm -r APPS.tar.gz
#rm -r dataset/apps_dataset
#echo "APPS done"

# Preprocess data

python get_data.py \
    config/config.yml