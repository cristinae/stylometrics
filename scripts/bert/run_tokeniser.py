# coding=utf-8

'''
   Conversion from plain text to a tokenised, tagged and padded input for BERT
   Adapted from https://github.com/vineetm/tfhub-bert/blob/master/bert_tfhub.ipynb
   Author: cristinae
   Date: 04.10.2020
'''

import tensorflow as tf
import tokenization
import os.path


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "input_file", None,
    "The input raw text file.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters
flags.DEFINE_string(
    "output_file", "output_bertTok.txt",
    "The text file with [CLS], [SEP], padding and indices.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")



def create_tokenizer(vocab_file='vocab.txt', do_lower_case=False):
    return tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)


def tokenize_sentence(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len-1:
        tokens = tokens[:max_seq_len-1]
    tokens.append('[SEP]')

    return tokens


def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = tokenize_sentence(sentence, tokenizer, max_seq_len)
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    #Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len-len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids


def convert_sentences_to_features(sentences, tokenizer, max_seq_len):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids




def main():

    # Commented because tokenisation can be independent of the model
    #if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    #   raise ValueError(
    #    "Cannot use sequence length %d because the BERT model "
    #    "was only trained up to sequence length %d" %
    #   (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tokenizer = create_tokenizer()

    input_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])
    input_mask = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])
    segment_ids = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, None])


    (dir_name, file_name) = os.path.split(FLAGS.input_file)
    (base_name, extension) = os.path.splitext(file_name)
    outFile=open(base_name+"_bertTok.txt", 'w')
    #sentences = ['New Delhi is the capital of India jhhnjui', 'The capital of India is Delhi']
    with open(FLAGS.input_file, 'r') as inFile:
         sentences = inFile.readlines() 
         for sentence in sentences:
             input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, FLAGS.max_seq_length)
             outFile.write(str(input_ids)[1:-1]+'\n')  #removing the square braquets    
    #input_ids_vals, input_mask_vals, segment_ids_vals = convert_sentences_to_features(sentences, tokenizer, FLAGS.max_seq_length)
    #toks = tokenize_sentence(sentences[0], tokenizer, 20)
    outFile.close()



if __name__ == "__main__":
   flags.mark_flag_as_required("input_file")
   flags.mark_flag_as_required("vocab_file")
   main()
