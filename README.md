# Multicam-Tracking-and-Counting



## Age and Gender Estimation
This is a Keras implementation of a CNN for estimating age and gender from a face image [1, 2].
In training, [the IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) is used.

- [Jun. 30, 2019] [Another PyTorch-based project](https://github.com/yu4u/age-estimation-pytorch) was released
- [Nov. 12, 2018] Enable Adam optimizer; seems to be better than momentum SGD
- [Sep. 23, 2018] Demo from directory
- [Aug. 11, 2018] Add age estimation sub-project [here](age_estimation)
- [Jul. 5, 2018] The UTKFace dataset became available for training.
- [Apr. 10, 2018] Evaluation result on the APPA-REAL dataset was added.

## Dependencies
- Python3.5+
- Keras2.0+
- scipy, numpy, Pandas, tqdm, tables, h5py
- dlib (for demo)
- OpenCV3

## Usage
### In client 
Run 
```
python3 client_0.py --server-ip [ip_server]
```
if you use IP camera. Please change IP address in client_0.py file 

### In Server
Run
```
python3 server.py
```
or if you need tracking + detect age and gender. Run
```
python3 server_agender.py
```
## References
[1] R. Rothe, R. Timofte, and L. V. Gool, "DEX: Deep EXpectation of apparent age from a single image," in Proc. of ICCV, 2015.

[2] R. Rothe, R. Timofte, and L. V. Gool, "Deep expectation of real and apparent age from a single image without facial landmarks," in IJCV, 2016.

[3] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond Empirical Risk Minimization," in arXiv:1710.09412, 2017.

[4] Z. Zhong, L. Zheng, G. Kang, S. Li, and Y. Yang, "Random Erasing Data Augmentation," in arXiv:1708.04896, 2017.

[5] E. Agustsson, R. Timofte, S. Escalera, X. Baro, I. Guyon, and R. Rothe, "Apparent and real age estimation in still images with deep residual regressors on APPA-REAL database," in Proc. of FG, 2017.
