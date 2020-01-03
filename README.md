# Music Score Converter
## Goal
This is an OCR model for musical scores I developed while at Insight Data Science. There are several programs that will convert images of sheet music to a computer editable format like [.musicxml](https://www.musicxml.com/) (this is essentially a standard way of encoding a musical score in xml). The best of these is probably [Audiveris](https://github.com/Audiveris), an [implementation of which](https://musescore.com/import) MuseScore itself has deployed. It's pretty good, but not perfect. It also uses classic techniques in computer vision. The goal of this project was to see if I could improve on it using a neural network.

The model works well on the training data, but definitely performs poorer than the Audiveris implementation outside of my training data. This is probably largely due to the fact that I used synthesized data to train the model.

## Model Overview
The goal is to construct a model which takes as input a .png file of a page of music and outputs a .musicxml file corresponding to the input image. The .musicxml file could then be edited on a computer. Suppose you had labeled data consisting of triples:
- a .png file of a page of music
- .musicxml file corresponding to that page of music
- bounding boxes for each measure.
Then you could train two neural nets: one which takes as input a page of music and crops out the measures, and one that takes as input an image of a measure and outputs the musicxml for that measure. You could combine the two neural nets to get a neural that takes as input a page of music and outputs the musicxml for each measure, which can be concatenated to get the musicxml for the page.

This is exactly what I did. For the image cropping net, I forked [a PyTorch implementation of Yolov3](https://github.com/ultralytics/yolov3). For the other net, I used a hybrid CNN/LSTM like that used in [pix2code](https://github.com/tonybeltramelli/pix2code). Raw .musicxml is a little too complicated for the model, so I created scripts that would convert .musicxml to and from a pseudocode like that used in the pix2code paper.

## Data
To get the labeled dataset, I synthesized random .musicxml files. See `./data/generate_sample.py` for the script used to generate the xml for individual measures. I used MuseScore, which can be run from the command line, to batch convert these files to .svg and .png files. I was able to use the metadata from the .svg files to extract data for bounding boxes for the.png files. See './data/generate_score_sample.py`.

## Process
I played around with some very simple data (one staff, any key, only quarter notes, no other symbols) and it worked very well. I then generalized to a more complicated dataset: piano scores, c major, any time signature, any whole through 32nd notes, no triplets, dynamics, dotted notes, no staccato, no slurs, no accents, no hairpins. If the model worked on this dataset, I would generalize to all musical markings. The model performed very well on the synthesized data (>90% note accuracy on a test set, pseudocode BLEU score of ~0.9). Unfortunately, it had some trouble generalizing to real samples. You can see it [deployed here](musictranscriber.com).