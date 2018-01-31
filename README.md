# cynical-selection
Allo-media data selection tool

This code implements the data selection method and algorithms proposed in Axelrod's paper [CYNICAL SELECTION OF LANGUAGE MODEL TRAINING DATA](https://arxiv.org/pdf/1709.02279.pdf), based on the paper's explanations and the Perl implementation Axelrod proposed [on github](https://github.com/amittai/cynical)

Comments in code and details on usage to come, but it's pretty simple right now.

## Basic usage

Say you have a (small) representative corpus (task.txt) and a (big) general one (unadapted.txt) and you want to select sentences from the big corpus that look like the small corpus ones.

Usage would be:
`./cynical-selection.py --task task.txt --unadapted unadapted.txt`

This will produce a `.jaded` file containing the selected sentences using the following tab-separated format:
`model score` `sentence score (penalty + gain)` `length penalty` `sentence gain` `sentence id (in the selection)` `sentence id (in the unadapted corpus)` `best word` `word gain` `sentence`.

See header of the script for available options, here is the two most important:

`batch`: essential with big corpora, allows to select more than one sentence at a time, see Axelrod's paper
`iterate`: iterate selection runs till no more than 10% of original size can be removed
