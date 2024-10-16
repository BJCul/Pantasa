# F7Xdict
* FlipVox pronunciation lexicon version 1.0
* [ARPAbet](http://en.wikipedia.org/wiki/Arpabet)-based pronunciation lexicon for Filipino spoken language processing
* License: Apache 2.0

For inquiries, please contact
Federico Ang (fmang AT ieee DOT org)

## Overview

* The core dictionary is patterned from the [CMU pronunciation dictionary](https://github.com/Alexir/CMUdict).  There are some minor differences:
    * spelled-letters are ALWAYS succeeded by an underscore (in the ARPA standard the last letter is NOT)
    * numerals are only allowed in the spelled-out case, and are also preceded by an underscore (e.g. `H_2_O_`)


* The non-core dictionary are entries derived from a model trained using the core dictionary on the non-official g2p-seq2seq product from Google.

* Details concerning the phone set will be included in later updates as these are currently fine-tuned along with the Kaldi-based ASR model.

If this is useful for your projects, please cite our work or reach out using the address above.  Thank you!
