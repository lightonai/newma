# -*- coding: utf-8 -*-
"""
Small video application of NEWMA: apply a one-dimensional NEWMA at each pixel, independently, for mean detection
(Psi = Id). Fast on CPU, GPU not required.
Requires numpy and opencv.

Place your web-cam in front of a relatively fixed background with a few moving objects.
"""
import numpy as np
import cv2
import onlinecp as ocp


print('press q to terminate') 
cap = cv2.VideoCapture(0)

big_Lambda = 0.3
small_lambda = big_Lambda/2
updt_threshold = big_Lambda/3
newma_obj = ocp.Newma(updt_coeff=small_lambda, updt_coeff2=big_Lambda, updt_coeff_thres=updt_threshold,
                      dist_func=lambda x, y: np.abs(x-y), thres_offset=1, thres_mult=1.5)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        res = newma_obj.update(np.sum(frame, axis=2)/3)  # sum over color channels and rescale
        frame[:, :, 1] = frame[:, :, 1] * (1 + 0.5 * res['result'])  # create and overlay change map
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if the job is finished
cap.release()
cv2.destroyAllWindows()
