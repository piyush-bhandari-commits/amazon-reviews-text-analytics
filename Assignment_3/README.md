# Report Assignment 3

The words most similar words to "good" are as follows: 

[('great', 0.9171791076660156),
 ('decent', 0.909214973449707),
 ('marvelous', 0.8900090456008911),
 ('terrific', 0.8766272068023682),
 ('fantastic', 0.876110851764679),
 ('nice', 0.8712728023529053),
 ('superb', 0.8590360879898071),
 ('phenomenal', 0.8577950596809387),
 ('wonderful', 0.854843020439148),
 ('impressive', 0.8462000489234924),
 ('respectable', 0.8354305624961853),
 ('snazzy', 0.8321412801742554),
 ('excellent', 0.8318706750869751),
 ('reasonable', 0.8300162553787231),
 ('exceptional', 0.8282021284103394),
 ('serviceable', 0.8267379999160767),
 ('fabulous', 0.8212142586708069),
 ('decently', 0.82084721326828),
 ('descent', 0.8165470361709595),
 ('lousy', 0.8153674006462097)]

The words most similar words to "bad" are as follows: 

[('awful', 0.8271024227142334),
 ('horrible', 0.8226331472396851),
 ('terrible', 0.8193413615226746),
 ('scary', 0.8152261972427368),
 ('funny', 0.8004640340805054),
 ('silly', 0.7770680785179138),
 ('obvious', 0.7703219652175903),
 ('shabby', 0.7695720195770264),
 ('horrendous', 0.7680036425590515),
 ('sucky', 0.766642689704895),
 ('iffy', 0.7603586912155151),
 ('horrific', 0.7587612867355347),
 ('nasty', 0.7576322555541992),
 ('stupid', 0.7558285593986511),
 ('enticing', 0.7466394901275635),
 ('wretched', 0.7465955018997192),
 ('overwhelming', 0.7455414533615112),
 ('strange', 0.7452280521392822),
 ('wtf', 0.7446393966674805),
 ('trivial', 0.7441580295562744)]
 
 ## Analysis
 
Considering words similar to 'good', we observe that the model determines the words similar to 'good' 
with a high accuracy. Exception will be the word 'lousy' which is unrelated, the reason for this can be
attributed to the fact that some reviews might have used negative terms to give a positive review in general.
For Example : 'The earlier version of the product was lousy, but this one is great'.

Considering words similar to 'bad', we observe that all the words are negative. The reason for finding the similar words 
correctly is less noise in the negative reviews. My intuition is for the result obtained is that the negative reviews 
contain less positive words to be associated with them.   
 
 
