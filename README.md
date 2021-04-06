# Simple Recommender with Attention
The code of modeling is taken from tensorflow recommenders.  
Inspired by Deep Cross Net and Self-Attention, I modified the Cross Layer, with key, query and value mechanism.  
Adding a SoftMax Layer & a Layer-Normalization Layer hurts the performance, so I remove them.  
# Performance
## Experiment Setup
* Dataset: MovieLens-100k
* Metric: Root Mean Square Error
* Environment: Tensorflow 2.4.1, Python 3.8.5, Windows 10 20H2, CPU Intel 1135G7, no GPU
## Experiment Result
| Evaluation | Deep Cross Net | Mine | Improvement |
|:-:|:-:|:-:|:-:|
| Time / step | 41ms | 35ms | -6ms |
| RMSE | 0.9239 | 0.9199 | -0.0040 |
## Future Work
An embedding size with 8 dimension achieves the best performance on my net (on MovieLens-20m, 6 dimension is the best, with RMSE 0.87).  
This is a significant reduction comparing to Deep Cross Net, where mine yields the performance gain.  
I am really curious about the reason, because generally larger dimensions means more information.  
