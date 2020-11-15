Eye Tracker
===
Idea
---
**Input**: WebCam image of face. \
**Output**: Pixel position of where the eyes are looking on the screen.

TODO
---
- ~~Tool to easily generate / annotate training data, i.e. pictures of *me* with + pixel location of where I'm looking.~~ Done, but can be improved.
- ~~Train or find pretrained model to find position of eyes.~~
- ~~Design and train model to predict cursor location based on picture of eye.~~
- Currently regressing position directly, which biases labels close to the mean --> switch to something more robust, like heatmap or anchor based supervision.
- Need some standard way to output of positions for external tools to read in real time.
- Improve README
    - How to generate face data
    - How to train
    - How to efficiently read output
