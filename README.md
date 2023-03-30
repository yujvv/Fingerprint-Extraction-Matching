# Fingerprint-Extraction-Matching
The important fingerprint minutiae features are the ridge endpoints (a.k.a. Terminations) and Ridge Bifurcations.

![image](https://user-images.githubusercontent.com/13918778/35665327-9ddbd220-06da-11e8-8fa9-1f5444ee2036.png)

The feature set for the image consists of the location of Terminations and Bifurcations and their orientations

# Note
use the code https://github.com/Utkarsh-Deshmukh/Fingerprint-Enhancement-Python to enhance the fingerprint image.
This program takes in the enhanced fingerprint image and extracts the minutiae features.

For the package of fingerprint minutiae extracted follow
`pip install fingerprint-feature-extractor`
`export QT_QPA_PLATFORM=offscreen`

Various methods of fingerprint matching are used in this project.
include
- Loop traversal minutiae point matching algorithm
- Minutia point matching algorithm based on KD-tree traversal
- Matching algorithm using OpenCV
- Matching algorithm using hash direct encryption
- Matching algorithm using block hash encryption
- Machine learning matching algorithm using tensor-network
- Matching Algorithm Using Match Scoring
- Matching algorithm using block hash+Match scoring
