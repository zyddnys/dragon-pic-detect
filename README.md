# Detecting Dragons
![dragon-detected.jpg](dragon-detected.jpg)
# How to use
First download all `template*.png`
```python
import cv2
from detect_sift import DragonDetector # or DragonDetectorFast
det = DragonDetector()
img = cv2.imread('test.png')
if det.is_dragon(img) :
    print('龙')
else :
    print('正常')
```
# Performance
Tuned for zero false positive.
## DragonDetector
2.4 images/sec
| Confusion matrix   |      Positive      |  Negative |
|----------|:-------------:|------:|
| Positive |  38 | 20 |
| Negative |    0   |   194 |
## DragonDetectorFast
37.45 images/sec
| Confusion matrix   |      Positive      |  Negative |
|----------|:-------------:|------:|
| Positive |  32 | 26 |
| Negative |    0   |   194 |

