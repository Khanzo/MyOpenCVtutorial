My OpenCV tutorial
-------------------
Dlib detector:
- dlib-19.8.1-cp36-cp36m-win_amd64.whl (http://biodev.ece.ucsb.edu:4040/bisque/dev/+simple/dlib)
- shape_predictor_68_face_landmarks.dat (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- pip install imutils

OpenCV:
pip install opencv_python-3.4.5+contrib-cp36-cp36m-win_amd64.whl (https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)

- images_example.py - операции с изображениями
- mouse_event.py - события мыши
- webcam.py - работа с веб-камерой
- facedetect.py - работа с детектом лица, глаз, рта
- facepoints.py - детект-точек лица

-----------------
Both SIFT and SURF authors require license fees for usage of their original algorithms.

I have done some research about the situation and here are the possible alternatives:

Keypoint detector:

Harris corner detector
Harris-Laplace - scale-invariant version of Harris detector (an affine invariant version also exists, presented by Mikolajczyk and Schmidt, and I believe is also patent free).
Multi-Scale Oriented Patches (MOPs) - athough it is patented, the detector is basically the multi-scale Harris, so there would be no problems with that (the descriptor is 2D wavelet-transformed image patch)
LoG filter - since the patented SIFT uses DoG (Difference of Gaussian) approximation of LoG (Laplacian of Gaussian) to localize interest points in scale, LoG alone can be used in modified, patent-free algorithm, tough the implementation could run a little slower
FAST
BRISK (includes a descriptor)
ORB (includes a descriptor)
KAZE - free to use, M-SURF descriptor (modified for KAZE's nonlinear scale space), outperforms both SIFT and SURF
A-KAZE - accelerated version of KAZE, free to use, M-LDB descriptor (modified fast binary descriptor)
Keypoint descriptor:

Normalized gradient - simple, working solution
PCA transformed image patch
Wavelet transformed image patch - details are given in MOPs paper, but can be implemented differently to avoid the patent issue (e.g. using different wavelet basis or different indexing scheme)
Histogram of oriented gradients
GLOH
LESH
BRISK
ORB
FREAK
LDB
Note that if you assign orientation to the interest point and rotate the image patch accordingly, you get rotational invariance for free. Even Harris corners are rotationally invariant and the descriptor may be made so as well.

Some more complete solution is done in Hugin, because they also struggled to have a patent-free interest point detector.
