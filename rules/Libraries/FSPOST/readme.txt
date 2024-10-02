/*******************************************************************************
 * 
 * FS-POST by: Matthew Phillip Go & Nicco Nocon, 2017
 * E-mail: noconoccin@gmail.com
 * 
 * Project Details:
 * Program: "Interdisciplinary Signal Processing for Pinoys (ISIP): Software 
 * Applications for Education (SAFE)" 
 * Project: Development of a Filipino Language Writing Tool
 * Institution: De La Salle University - Manila
 * Project Manager: Allan B. Borra
 * 
 * Filipino Stanford Part-of-Speech Tagger (FSPOST) utilizes the Stanford
 * Part-of-Speech Tagger. A Filipino tagger model was developed in order to tag
 * Filipino sentences. The model was trained using the following feature sets: 
 * left5words, distsim, prefix(6), and prefix(2,1), using the owlqn2 
 * optimization. For the tagger's delimiter, it uses the vertical bar '|' symbol. 
 * The Stanford POS Tagger can be accessed in this website: 
 * https://nlp.stanford.edu/software/tagger.shtml and the MGNN tagset (list of 
 * POS tags) used can be accessed here: http://goo.gl/dY0qFe
 * 
 * For instructions how to use the Stanford POS Tagger and the Filipino tagger 
 * model, read through the tagger's Java documentation. 
 * Note: The tagger may also be used in other languages as seen in the Stanford 
 * POS Tagger's homepage link above.
 * 
 ******************************************************************************/

 Requirements:
	7MB Hard Disk Drive Memory
	Java
	
 Instructions:
	1) Import the Stanford POS Tagger jar (Filipino-Stanford-Part-of-Speech-Tagger\stanford-postagger.jar) on your program.
	2) Move the tagger model file (ilipino-Stanford-Part-of-Speech-Tagger\filipino-left5words-owlqn2-distsim-pref6-inf2.tagger) 
	   on your project folder.
	3) Enter the tagging function call utilizing the tagger model (tutorials at https://nlp.stanford.edu/software/tagger.shtml)
	4) Run!
	
 References: 
	
 - Kristina Toutanova and Christopher D. Manning. 2000. Enriching the Knowledge Sources Used in a Maximum Entropy Part-of-Speech Tagger. In 
   Proceedings of the Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora (EMNLP/VLC-2000), pp. 63-70.
	
 - Kristina Toutanova, Dan Klein, Christopher Manning, and Yoram Singer. 2003. Feature-Rich Part-of-Speech Tagging with a Cyclic Dependency Network. 
   In Proceedings of HLT-NAACL 2003, pp. 252-259.
	
 - Matthew Phillip Go and Nicco Nocon. 2017. Part-of-Speech Tagging for Filipino. Unpublished.