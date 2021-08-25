# TEFE
TEFE - TimeBankPT Event Frame Extraction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FORMAS/TEFE/blob/main/notebook/colab-tefe.ipynb)

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/andersonsacramento/tefe)


# DESCRIPTION

TEFE is a closed domain event extractor system for sentences in the Portuguese language. It extracts events from sentences, which means that it does event detection (i.e., event trigger identification and classification), and argument role prediction (i.e., argument identification and role classification). The event types are based on the typology of the FrameNet project (BAKER; FILLMORE; LOWE, 1998). The models were trained on an enriched TimeBankPT (COSTA; BRANCO,2012) corpus.

The system outputs the event extractions in the following Json format:
```json
[{
  "trigger":    { 
                 "text":   "disse",
	         "start":  58,
	         "end":    63,
	        },  
  "event_type": "Statement",
  "arguments":  [{
                  "role":  "Speaker",
	          "text":  "presidente",
	          "start": 66,
	          "end":   76
		  },
		  ...
		]    
  },
  ...
]
  
```
Currently, in this repository, 5 diferent trained models are avaiable to execution: 0, 100, 0-0, 100-0, 100-100, which respectively correspond to: 514 event types (ET) and 1936 argument roles (AR), 7 ET and 93 AR, 214 ET and 477 AR, 5 ET and 42 AR, and 5 ET and 12 AR.

# Local Execution

## Prerequisites

1. Download and place the BERTimbau Base (SOUZA; NOGUEIRA;LOTUFO, 2020) model and vocabulary file:
    ```bash
    $ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/bert-base-portuguese-cased_tensorflow_checkpoint.zip
	```
	```bash
	$ wget https://neuralmind-ai.s3.us-east-2.amazonaws.com/nlp/bert-base-portuguese-cased/vocab.txt
	```
	Then unzip and place it in the the models directory as follows:
	```
	├──models
	|      └── BERTimbau
	|               └── bert_config.json
	|               └── bert_model.ckpt.data-00000-of-00001
	|               └── bert_model.ckpt.index
	|               └── bert_model.ckpt.meta
	|               └── vocab.txt
	|
	|...
	```

2. Install the packages.
   ```bash
   $ pip install -r requirements.txt
   ```

<!-- 3. Download and place all the other models (i.e., 0, 0-0, 100-0, 100-100): -->
<!--    * [Download link](https://drive.google.com/file/d/1isdiyUu5sXkS8DXdEBjE_pfi-ErjnutQ/view?usp=sharing) -->
<!--    * Then unzip and place it in the models directory as follows: -->
<!-- 	``` -->
<!-- 	├──models -->
<!-- 	|     └── blstme_0_0.h5 -->
<!-- 	|     └── blstme_100_100.h5 -->
<!-- 	|     └── blstme_100.h5 -->
<!-- 	|     └── blstmea_0.h5 -->
<!-- 	|     └── blstmeat2_100_0.h5 -->
<!-- 	| -->
<!-- 	|... -->
<!-- 	``` -->


# OPTIONS
    -h, --help                           Print this help text and exit
	--sentence  SENTENCE                 Sentence string to extract events from
	--dir   INPUT-DIR OUTPUT-DIR         Extract events from files of input directory
		                                 (one sentence per line) and write output json
										 files on output directory.
    --model  ID                          Identifier of models available: 0, 100, 0-0, 100-0 or 
	                                     100-100. The default model is 100


## EVENT EXTRACTION FROM A DIRECTORY OF FILES
The text files in the input directory are expected to have the format:

    * all text files end with the extension .txt
    * sentences are separated by newlines
	
```bash
$ python3 src/tefe.py --dir /tmp/input-dir /tmp/output-dir
```
## EVENT EXTRACTION FROM A SENTENCE

```bash
$ python3 src/tefe.py --sentence 'A Petrobras aumentou o preço da gasolina para 2,30 reais, disse o presidente.'
```
## How to cite this work

Peer-reviewed accepted paper:

10th Brazilian Conference on Intelligent Systems (BRACIS)

* Sacramento A. ; Souza M. . Joint Event Extraction with Contextualized Word Embeddings for the Portuguese 
Language.
