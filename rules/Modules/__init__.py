# project/Modules/__init__.py
from rules.Modules.Tokenizer import split_sentences
from rules.Modules.POSDTagger import pos_tag as pos_dtag
from rules.Modules.POSRTagger import pos_tag as pos_rtag
from rules.Modules.Lemmatizer import lemmatize_sentence 

_all_= ['Tokenizer','POSDTagger','POSRTagger','Lemmatizer']