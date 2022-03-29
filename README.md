# TPNMS

**Code and models for AAAI 2021 paper: *Temporal Pyramid Network for Pedestrian Trajectory Prediction with  Multi-Supervision***

### Environment

 - Python 3.8
 - pytorch 1.11.0
 - cuda 11.3
 - Ubuntu 20.04
 - RTX 3090
 - Please refer to the "requirements.txt" file for more details.

### Usage  
To test the model, run: 
```bash
scripts/evaluate_model.py
```
To train the model, run: 
```bash
scripts/train_TPN_P.py
```



### Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{liang2021temporal,
  title={Temporal Pyramid Network for Pedestrian Trajectory Prediction with Multi-Supervision},
  author={Liang, Rongqin and Li, Yuanman and Li, Xia and Tang, Yi and Zhou, Jiantao and Zou, Wenbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={3},
  pages={2029--2037},
  year={2021}
}
```

### Contact

If you encounter any issue when running the code, please feel free to reach us either by creating a new issue in the github or by emailing

+ 1810262064@email.szu.edu.cn
