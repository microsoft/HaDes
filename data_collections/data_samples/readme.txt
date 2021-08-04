A token-level reference-free hallucination detection dataset (HaDeS)


One instance (JSON format) per line in the file. The task is a binary classification (0 for "hallucination"/ 1 for "not hallucination") for the specified word/span in the given context (the "replaced" text). The "replaced_ids" tuple indicates the beginning and ending positions in the text. For "train.txt" and "valid.txt", the "hallucination" label (0/1) is provided while for all instances in "test.txt", we do not provide the gold labels, i.e. the "hallucination" label is set as -1.

For better visualization, we highlight the target word/span with a marker, e.g. "===made it===" in the "replaced" text, but you should remove marker while training or predicting.

One example instance:
{"replaced": "baxter made his wallaby debut against the all blacks during the 2003 bledisloe cup . he ===made it=== his 100th test cap against france during the 2007 rugby world cup , scoring his one and only test try . he has become only the second australian player to reach the milestone along with former nsw waratahs prop ewen mckenzie . he scored his first super rugby try on his first appearance for the nsw waratahs . he is the third most capped australian , only behind mark de villiers and ben alexander . baxter retired from the sport in 2011 to pursue a career in business . in 2011 , he was appointed as vice - president of the rugby union .", "replaced_ids": [16, 17], "hallucination": 0}

Please refer to https://github.com/tyliupku/HaDeS for more details.