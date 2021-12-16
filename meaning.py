import numpy as np
import torch

def meaning(imag, rd):

    n = rd

    zer = torch.zeros_like(imag)
    x = imag.shape[1]
    y = imag.shape[2]

    win_x = x//n
    win_y = y//n

    step_x = x//win_x
    step_y = y//win_y
    mean_type = torch.median

    for i in range(step_x-1):
        for j in range(step_y-1):
            zer[0,i*win_x:(i+1)*win_x, j*win_y:(j+1)*win_y] += mean_type(imag[0,i*win_x:(i+1)*win_x, j*win_y:(j+1)*win_y])
            zer[1,i*win_x:(i+1)*win_x, j*win_y:(j+1)*win_y] += mean_type(imag[1,i*win_x:(i+1)*win_x, j*win_y:(j+1)*win_y])


    for i in range(step_x - 1):
        zer[0, i*win_x:(i+1)*win_x, (step_y-1)*win_y:] += mean_type(imag[0, i*win_x:(i+1)*win_x, (step_y-1)*win_y:])
        zer[1, i*win_x:(i+1)*win_x, (step_y-1)*win_y:] += mean_type(imag[1, i*win_x:(i+1)*win_x, (step_y-1)*win_y:])


    for j in range(step_y-1):
        zer[0, (step_x-1)*win_x:, j*win_y:(j+1)*win_y] += mean_type(imag[0, (step_x-1)*win_x:, j*win_y:(j+1)*win_y])
        zer[1, (step_x-1)*win_x:, j*win_y:(j+1)*win_y] += mean_type(imag[1, (step_x-1)*win_x:, j*win_y:(j+1)*win_y])


    zer[0, (step_x-1)*win_x:, (step_y-1)*win_y:] += mean_type(imag[0, (step_x-1)*win_x:, (step_y-1)*win_y:])
    zer[1, (step_x-1)*win_x:, (step_y-1)*win_y:] += mean_type(imag[1, (step_x-1)*win_x:, (step_y-1)*win_y:])

    return zer