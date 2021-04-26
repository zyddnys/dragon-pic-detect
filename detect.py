import numpy as np
import cv2

dragon_cascade = cv2.CascadeClassifier('haarcascade_dragon.xml')

img = cv2.imread('dragon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dragons = dragon_cascade.detectMultiScale(gray, 1.01, 8)
for (x, y, w, h) in dragons :
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imwrite('dragon-detected.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
