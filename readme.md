# Annotation Game
In this game, participants are required to annotate data according to some ___target concept___ in `T` rounds.

<img src="imgs/task illustration.png" width = "200" align=left />

Each round consists of two phases:  
__Test__:  
    _The participant will be required to annotate `K` instances from the dataset._  
__Teaching__:   
    _`M` annotation examples from experts will be show to the participant._  
    
Every participant will be paid \$0.01 for the first round, and get ___performance-contingent bonus___ \$0.02 each round after. 

### Please do as well as you can!
We will qualify you for bouns in the test phase, and suggest you some expert labels for help in the teaching phase.  
The bouns is given accordingly to your __relative performance__ (a.k.a improvement) rather than your absolute performance (a.k.a. accuracy).  
Therefore, the best way to maximize your gains is to do the best in each round! __You would earn as much as experts do!__

--------
## Task:  "ピンク"
Please annotate all the "ピンク" instances with "1" and others with "0".

<img src="imgs/task illustration 2.png" width = "320" align=left />

Features $x = (f_1, f_2, f_3)$:  
- `Shape` $[f_1]$: `1` = `triangle`, `0` = `circle`. 
- `Border` $[f_2]$: `1` = `real border`, `0` = `dotted border`.
- `Color` $[f_3]$: `1` = `pink`, `0` = `bule`.

Hypothesis Space $\mathcal H = \{h_1, h_1', h_2, h_2', h_3, h_3'\}$
-  $h_1(x) = f_1$ (`ピンク is triangle`) and $h_1'(x) = 1 - f_1$ (`ピンク is circle`)
- $h_2(x) = f_2$ (`ピンク is real`) and $h_1'(x) = 1 - f_2$ (`ピンク is dotted`)
- $h_3(x) = f_3$ (`ピンク is pink`) and $h_1'(x) = 1 - f_3$ (`ピンク is bule`)