# RCNN Adaptation for XLing CQA

This repository contains the code for the cross lingual programming CQA experiments on AskUbuntu and StackOverflow datasets.

The dictory rcnn is cloned from [TaoLei/Rcnn](https://github.com/taolei87/rcnn). And come minor modifications have been made in the code for training only on titles, applying translations, running the scripts, and analyzing the predictions.


## How to Run

The original RCNN project uses an old Theano version that requires cuda 8. So you need to set some environment variables first, similar to the following:

```
export PATH="/usr/local/cuda-8.0/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export PYTHONPATH=<path to rcnn/code>
cd rcnn/code
```

### Pretraining

```
$ cd pt
$ CUDA_VISIBLE_DEVICES="0" THEANO_FLAGS='device=gpu,floatX=float32' python main.py --corpus <path_to_data_folder>/text_tokenized.txt.gz --embeddings <path_to_data_folder>/vector/vectors_pruned.200.txt.gz --train <path_to_data_folder>/train_random.txt --dev <path_to_data_folder>/dev.txt --test <path_to_data_folder>/test.txt --heldout <path_to_data_folder>/heldout.txt --dropout 0.1 -d 400 --model <model_name_to_save>.pkl.gz
``` 

### Training

```
$ cd qa
$ CUDA_VISIBLE_DEVICES="0" THEANO_FLAGS='device=gpu,floatX=float32' python main-title.py --corpus <path_to_data_folder>/text_tokenized.txt.gz --embeddings <path_to_data_folder>/vector/vectors_pruned.200.txt.gz --train <path_to_data_folder>/train_random.txt --dev <path_to_data_folder>/dev.txt --test <path_to_data_folder>/test.txt --dropout 0.1 -d 400 --save_model <model_name_to_save>.pkl.gz --load_pretrain ../pt/<name_of_saved_model_or_pretrained_model>.pkl.gz --do_train 1
``` 


### Evaluate a model
```
$ cd qa_or_qa-mixed-full 
$ CUDA_VISIBLE_DEVICES="0" THEANO_FLAGS='device=gpu,floatX=float32' python main-title.py --corpus <path_to_data_folder>/text_tokenized.txt.gz --embeddings <path_to_data_folder>/vector/vectors_pruned.200.txt.gz --train <path_to_data_folder>/train_random.txt --dev <path_to_data_folder>/dev.txt --test <path_to_data_folder>/test.txt --dropout 0.1 -d 400 --save_model <model_name_to_save>.pkl.gz --load_pretrain ../pt/<name_of_saved_model_or_pretrained_model>.pkl.gz --do_evaluate 1
```


### Evaluate with Translations

```
CUDA_VISIBLE_DEVICES="0" THEANO_FLAGS='device=gpu,floatX=float32' python main-title.py --corpus <path_to_data_folder>/text_tokenized.txt.gz --embeddings <path_to_data_folder>/vector/vectors_pruned.200.txt.gz --train <path_to_data_folder>/train_random.txt --dev <path_to_data_folder>/dev.txt --test <path_to_data_folder>/test.txt --translations <path_to_data_folder>-gt-gt-full/text_tokenized.txt.gz --translatable_ids <path_to_data_folder>/test.txt  --layer rcnn --dropout 0.1 -d 400 --batch_size 256 --margin 0.1 --max_epoch 100 --load_pretrain the_trained_rcnn_model.pkl.gz --do_evaluate 1
```

Similar changes need to be performed for training, e.g., when you would like to train on paraphrases obtained by backtranslating text (set the --translatable_ids and --translations accordingly).