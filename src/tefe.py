import sys
import os
import numpy as np
import re
import argparse
import json
import glob

from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
import tensorflow as tf
from tensorflow.keras.models import load_model


BERTIMBAU_MODEL_PATH = 'models/BERTimbau/'
EMBEDDING_ID = 'last_hidden'


RUN_CONFIGS = {
        '0':       {'model':        'models/blstmea_0.h5',
                    'events-types': 'res/events_by_pos_types_514.json',
                    'args-types':   'res/args_by_pos_types_1936.json'},
        '100':     {'model':        'models/blstme_100.h5',
                    'events-types': 'res/events_by_pos_types_7.json',
                    'args-types':   'res/args_by_pos_types_93.json'},
        '0-0':     {'model':        'models/blstme_0_0.h5',
                    'events-types': 'res/events_by_pos_types_214.json',
                    'args-types':   'res/args_by_pos_types_477.json'},
        '100-0':   {'model':        'models/blstmeat2_100_0.h5',
                    'events-types': 'res/events_by_pos_types_5.json',
                    'args-types':   'res/args_by_pos_types_42.json'},
        '100-100': {'model':        'models/blstme_100_100.h5',
                    'events-types': 'res/events_by_pos_types_5.json',
                    'args-types':   'res/args_by_pos_types_12.json'}}

DEFAULT_RUN_CONFIG = '100'

def tokenize_and_compose(text):
        tokens = tokenizer.tokenize(text)
        text_tokens = []
        for i, token in enumerate(tokens):
            split_token = token.split("##")
            if len(split_token) > 1:
                token = split_token[1]
                text_tokens[-1] += token
            else:
                text_tokens.append(token)
        if len(text_tokens[1:-1]) == 1:
          return text_tokens[1]
        else:
          return text_tokens[1:-1]


def compose_token_embeddings(sentence, tokenized_text, embeddings):
        tokens_indices_composed = [0] * len(tokenized_text)
        j = -1
        for i, x in enumerate(tokenized_text):
            if x.find('##') == -1:
                j += 1
            tokens_indices_composed[i] = j
        word_embeddings = [0] * len(set(tokens_indices_composed))
        j = 0
        for i, embedding in enumerate(embeddings):
            if j == tokens_indices_composed[i]:
                word_embeddings[j] = embedding
                j += 1
            else:
                word_embeddings[j - 1] += embedding
        return word_embeddings

def extract(text, options={'sum_all_12':True}, seq_len=512, output_layer_num=12):
        features = {k:v for (k,v) in options.items() if v}
        tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first = text, max_len = seq_len)
        predicts = model_bert.predict([np.array([indices]), np.array([segments])])[0]
        predicts = predicts[1:len(tokens)-1,:].reshape((len(tokens)-2, output_layer_num, 768))

        for (k,v) in features.items():
            if k == 'sum_all_12':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts.sum(axis=1))
            if k == 'sum_last_4':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-4:,:].sum(axis=1))
            if k == 'concat_last_4':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-4:,:].reshape((len(tokens)-2,768*4)))
            if k == 'last_hidden':
                features[k] = compose_token_embeddings(text, tokens[1:-1], predicts[:,-1:,:].reshape((len(tokens)-2, 768)))
        return features



def get_sentence_original_tokens(sentence, tokens):
        token_index = 0
        started = False
        sentence_pos_tokens = []
        i = 0
        while i < len(sentence):
                if sentence[i] != ' ' and not started:
                        start = i
                        started = True
                if sentence[i] == tokens[token_index] and started:
                        sentence_pos_tokens.append(sentence[i])
                        started = False
                        token_index += 1
                elif i<len(sentence) and (sentence[i] == ' ' or tokenize_and_compose(sentence[start:i+1]) == tokens[token_index] ) and started:
                        sentence_pos_tokens.append(sentence[start:i+1])
                        start = i+1
                        started = False
                        token_index += 1
                i += 1
        return sentence_pos_tokens


def get_text_location(text, arg, start_search_at=0):
        text = text.lower()
        arg = arg.lower()
        pattern = re.compile(r'\b%s\b' % arg)
        match = pattern.search(text, start_search_at)
        if match:
                return (match.start(), match.end())
        else:
                return (-1, -1)


def get_args_from_labels(label_args, is_arp=True):
        args = []
        cur_arg = []
        started_arg = False
        fn_normalize_label = lambda cur_label : cur_label[-1] if cur_label[-1] <= len(args_types) else cur_label[-1] - len(args_types)
        for i,label in enumerate(label_args):
                if not started_arg and label != 0 and label <= len(args_types):
                        cur_arg.append((i, label if is_arp else 1))
                        started_arg = True
                elif started_arg and label != 0 and label > len(args_types):
                        last_label = fn_normalize_label(cur_arg[-1])
                        if label-len(args_types) != last_label and is_arp:
                                cur_arg = []
                                started_arg = False
                        else:
                                cur_arg.append((i,label if is_arp else 1))
                elif started_arg and label == 0:
                        args.append(tuple(cur_arg))
                        cur_arg = []
                        started_arg = False
                elif started_arg and  label <= len(args_types):
                        args.append(tuple(cur_arg))
                        cur_arg = []
                        cur_arg.append((i, label if is_arp else 1))
                        started_arg = True
        if cur_arg:
                args.append(tuple(cur_arg))
        return args


def extract_events(text, feature_option):
        text_tokens = get_sentence_original_tokens(text, tokenize_and_compose(text))
        features = extract(text, {feature_option:True})[feature_option]
        embeddings = np.array(features).reshape((len(text_tokens), 768))
        sentence_embeddings = np.zeros((1,128,768))
        sentence_embeddings[0,:len(text_tokens)] = embeddings
        predictions = [model.predict([e.reshape((1, 768)), sentence_embeddings]) for e in embeddings]
        positions = list(filter((lambda i: i>= 0 and i < len(text_tokens)), [pos for (pos, (pred_ed, pred_args)) in enumerate(predictions) if np.argmax(pred_ed) != 0]))
        output = []
        if len(positions) > 0:
                start_at = sum([len(token) for token in text_tokens[:positions[0]]])
        for pos in positions:
                loc_start, loc_end = get_text_location(text, text_tokens[pos], start_at)
                start_at = loc_end
                args_preds =  [np.argmax(predictions[pos][1][0,i,:]) for i in range(predictions[pos][1].shape[1]) if i < len(text_tokens)]
                start_arg_search = 0
                args_event = []
                event_type = events_types[str(np.argmax(predictions[pos][0]))]
                for arg_tokens in get_args_from_labels(args_preds):
                        first_arg_token = arg_tokens[0]
                        last_arg_token = arg_tokens[-1]
                        pattern = re.compile(r'\b%s\b' % '\s*'.join([text_tokens[arg_token[0]] for arg_token in arg_tokens]))
                        match = pattern.search(text, start_arg_search)
                        if match:
                                arg_type = args_types[str(first_arg_token[1])]
                                if str(arg_type['id']) in event_type['args']:
                                        args_event.append({'role':arg_type['name'],
                                                           'text': text[match.start():match.end()],
                                                           'start': match.start(),
                                                           'end': match.end()
                                                           })
                           
                                start_arg_search = match.end()
                output.append({'trigger':{
                        'text': text[loc_start:loc_end],
                        'start': loc_start,
                        'end' : loc_end},
                               'arguments':args_event,
                               'event_type': event_type['name']
                               })
        return output



def load_bertimbau_model():    
        global tokenizer
        global model_bert
        
        paths = get_checkpoint_paths(BERTIMBAU_MODEL_PATH)

        model_bert = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, seq_len=512, output_layer_num=12)

        token_dict = load_vocabulary(paths.vocab)
        tokenizer = Tokenizer(token_dict)

def load_tefe_model():
        global model
        global events_types
        global args_types

        events_types, args_types = load_events_args_info()
        model = load_model(RUN_CONFIGS[model_config]['model'])
        return model

def load_events_args_info():
        events_types, args_types = {}, {}

        with open(RUN_CONFIGS[model_config]['events-types'], 'r') as read_content:        
                events_types = json.load(read_content)
                
        with open(RUN_CONFIGS[model_config]['args-types'], 'r') as read_content:        
                args_types = json.load(read_content)                

        return events_types, args_types



def extract_from_files(input_path, output_path):
        for filepathname in glob.glob(f'{input_path}*.txt'):
                extractions = []
                for line in open(filepathname):
                        line = line.strip()
                        print(line)
                        extractions.append(extract_events(line, EMBEDDING_ID))

                filename = filepathname.split('.txt')[0].split(os.sep)[-1]
                with open(f'{output_path}{filename}.json', 'w')  as outfile:
                        json.dump(extractions, outfile)
                print(f'{filename}')


def extract_events_from(input_path, output_path):
        run_extraction_context(lambda : extract_from_files(input_path, output_path))
        

def extract_events_from_sentence(sentence):
        sentence = sentence.strip()
        run_extraction_context(lambda : print(extract_events(sentence, EMBEDDING_ID)))
        

def run_extraction_context(run_extraction_func):                        
        if len(tf.config.list_physical_devices('GPU')) > 0:
                with tf.device('/GPU:0'):
                        load_bertimbau_model()
                        load_tefe_model()
                        run_extraction_func()
        else:
                with tf.device('/cpu:0'):
                        load_bertimbau_model()
                        load_tefe_model()
                        run_extraction_func()
                        

if __name__ == '__main__':
        global model_config
        model_config = DEFAULT_RUN_CONFIG
        
        parser = argparse.ArgumentParser(description="TEFE - TimeBankPT Event Frame Extraction")
        parser.add_argument('--sentence', type=str,
                            help='sentence string to be extracted the events')
        parser.add_argument('--dir',
                            nargs=2,
                            help='relative path to directory with files of sentences to be extracted')
        parser.add_argument('--model',
                            default=DEFAULT_RUN_CONFIG,
                            help='model name to process the inputs')
        

        args = parser.parse_args()

        if args.model and args.model in RUN_CONFIGS:
                model_config = args.model

        if args.dir:
                input_dir, output_dir = args.dir
                if input_dir and output_dir and os.path.exists(input_dir) and os.path.exists(output_dir):
                        extract_events_from(input_dir, output_dir)
        if args.sentence:
                extract_events_from_sentence(args.sentence)
    

    

