def handle_nang_ng(text, pos_tags):
    words = text.split()
    corrected_words = []
    
    for i, word in enumerate(words):
        if word == "nang":
            # Handle "nang" as a synonym for "noong" (RBW)
            if pos_tags[i] == 'RBW':  
                corrected_words.append("noong")
            
            # Handle "nang" as a conjunction (CCB, CCT)
            elif pos_tags[i] in ['CCB', 'CCT']:
                corrected_words.append(word)  # Keep "nang" as conjunction
            
            # Handle "nang" as a ligature (CCP)
            elif pos_tags[i] == 'CCP' and i > 0:
                prev_pos = pos_tags[i - 1]
                if prev_pos in ['RB', 'VB', 'JJ']:  # Connecting adverb of manner/intensity to verb/adjective
                    corrected_words.append(word)
                else:
                    corrected_words.append("nang")
        
        elif word == "ng":
            # Handle "ng" as a ligature (CCP or CCB)
            if pos_tags[i] in ['CCP', 'CCB']:
                corrected_words.append(word)  # Keep "ng"
            else:
                corrected_words.append(word)
        
        else:
            corrected_words.append(word)
    
    return ' '.join(corrected_words)
