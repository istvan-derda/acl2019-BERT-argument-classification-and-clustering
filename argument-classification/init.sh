#! /bin/bash

echo Downloading pretrained BERT...
wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip

echo unpacking...
unzip argument_classification_ukp_all_data.zip
mv argument_classification_ukp_all_data/pytorch_model.bin bert_output/argument_classification_ukp_all_data/
rm -rf argument_classification_ukp_all_data argument_classification_ukp_all_data.zip 

echo Downloading args.me corpus...
wget --no-check-certificate https://files.webis.de/data-in-progress/data-research/arguana/touche/touche22/args_processed_04_01.tar.gz

echo unpacking...
tar -xf args_processed_04_01.tar.gz
rm args_processed_04_01.tar.gz
