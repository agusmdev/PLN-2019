#! /bin/bash
echo "Training Model 1-Gram"
python3 languagemodeling/scripts/train.py -n 1 -o trained_models/1gram -m ngram
echo "Training Model 2-Gram"
python3 languagemodeling/scripts/train.py -n 2 -o trained_models/2gram -m ngram
echo "Training Model 3-Gram"
python3 languagemodeling/scripts/train.py -n 3 -o trained_models/3gram -m ngram
echo "Training Model 4-Gram"
python3 languagemodeling/scripts/train.py -n 4 -o trained_models/4gram -m ngram
echo "Training Model 1-AddOneNGram"
python3 languagemodeling/scripts/train.py -n 1 -o trained_models/addone1gram -m addone
echo "Training Model 2-AddOneNGram"
python3 languagemodeling/scripts/train.py -n 2 -o trained_models/addone2gram -m addone
echo "Training Model 3-AddOneNGram"
python3 languagemodeling/scripts/train.py -n 3 -o trained_models/addone3gram -m addone
echo "Training Model 4-AddOneNGram"
python3 languagemodeling/scripts/train.py -n 4 -o trained_models/addone4gram -m addone
echo "Training Model 1-InterpolatedNGram"
python3 languagemodeling/scripts/train.py -n 1 -o trained_models/inter1gram -m inter
echo "Training Model 2-InterpolatedNGram"
python3 languagemodeling/scripts/train.py -n 2 -o trained_models/inter2gram -m inter
echo "Training Model 3-InterpolatedNGram"
python3 languagemodeling/scripts/train.py -n 3 -o trained_models/inter3gram -m inter
echo "Training Model 4-InterpolatedNGram"
python3 languagemodeling/scripts/train.py -n 4 -o trained_models/inter4gram -m inter
echo "Training Model 1-BackOffNGram"
python3 languagemodeling/scripts/train.py -n 1 -o trained_models/backoff1gram -m backoff
echo "Training Model 2-BackOffNGram"
python3 languagemodeling/scripts/train.py -n 2 -o trained_models/backoff2gram -m backoff
echo "Training Model 3-BackOffNGram"
python3 languagemodeling/scripts/train.py -n 3 -o trained_models/backoff3gram -m backoff
echo "Training Model 4-BackOffNGram"
python3 languagemodeling/scripts/train.py -n 4 -o trained_models/backoff4gram -m backoff