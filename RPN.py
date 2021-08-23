import cv2
import sys

img_path = "3_idiots.png"

image = cv2.imread(img_path)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# high recall selective search
ss.switchToSelectiveSearchQuality()

rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

numShowRects = 100
imOut = image.copy()

for i, rect in enumerate(rects):
  if (i < numShowRects):
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
  else:
    break

cv2.imshow("Output", imOut)

cv2.imwrite("3_idiots_SS_out.png", imOut)
cv2.imshow("Original", image)
cv2.waitKey(0)
