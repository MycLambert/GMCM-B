#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
from scipy.integrate import dblquad  # dblquad用于二重积分
import matplotlib.pyplot as plt
from scipy import optimize
import sympy


def func(x, a, b, c, d, e):
    f = a * x * x * x * x + b * x * x * x + c * x * x + d * x + e
    return f.ravel()

def plot_display(x0, y0, figname, line):
    # 曲线拟合与绘制
    display = plt.figure(figname)
    #plt.scatter(x0[:], y0[:], 25, "red")
    plt.title(figname)
    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    ''''
    plt.title('Inversion')
    plt.ylabel('Error')
    plt.xlabel('Count')
    '''
    xdata = np.array(x0)
    #print xdata

    #popt, pcov = optimize.curve_fit(func, xdata, y0)
    # popt数组中，三个值分别是待求参数a,b,c
    #y2 = [func(i, popt[0], popt[1], popt[2], popt[3], popt[4]) for i in x0]
    #plt.logy = True
    ax, = plt.semilogy(x0, y0)
    #ax, = plt.plot(x0, y0)
    plt.ylim(0.000000001, 1)
    #plt.ylim(0.1, 1)
    #plt.xlim(0, 25)
    #ax1, = plt.plot(x0, y0)

    return ax
    #print popt

def cal_func(SNRdB, mode = '16QAM'):
    if mode == 'QPSK':
        return cal_func_QPSK(SNRdB)
    if mode == '8QAM':
        return cal_func_8QAM(SNRdB)
    if mode == '16QAM':
        return cal_func_16QAM(SNRdB)
    if mode == '16QAM_change':
        return cal_func_16QAM_change(SNRdB, 0.1932, 0.0284, 10 ** (-6))

def cal_func_QPSK(SNRdB):
    sigma = math.sqrt(2) / 2 * (10 ** (-SNRdB / 20))
    P11_10 = cal_integ(sigma, 0.5, float('inf'), -0.5, float('inf'))
    P11_00 = cal_integ(sigma, 0.5, float('inf'), -float('inf'), -0.5)
    Ne = (2 * P11_10 + 2 * P11_00) / 2
    return Ne

def cal_func_8QAM(SNRdB):
    sigma = (3 + math.sqrt(3)) / 4 * (10 ** (-SNRdB / 20))
    P000_101 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -float('inf'), -1, lambda x: -x, lambda x: float('inf'))[0] \
        + dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                  , -1, -0.5, lambda x: -0.5 * (x - 1), lambda x: float('inf'))[0] \
        + dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                  , 0.5, 2, lambda x: 0.5 * x, lambda x: float('inf'))[0] \
        + dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                  , 2, float('inf'), lambda x: x - 1, lambda x: float('inf'))[0]


    P000_001 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0.5, 1 + math.sqrt(3) / 6, lambda x: -0.5, lambda x: 0.5 * x)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 1 + math.sqrt(3) / 6, 2, lambda x: math.sqrt(3) * x - (math.sqrt(3) + 1) / 2, lambda x: 0.5 * x)[0]

    P000_111 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -float('inf'), -math.sqrt(3) / 2 - 1, lambda x: -x, lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -math.sqrt(3) / 2 - 1, -0.5, lambda x: - 1 / math.sqrt(3) * x + 0.5 + math.sqrt(3) / 6, lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -0.5, math.sqrt(3) / 2, lambda x: 1 / math.sqrt(3) * x + 0.5 + math.sqrt(3) / 6, lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , math.sqrt(3) / 2, float('inf'), lambda x: x + 1, lambda x: float('inf'))[0]

    P000_011 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0.5, 1 + math.sqrt(3) / 6, lambda x: -1 / math.sqrt(3) * x - 1, lambda x: -0.5)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 1 + math.sqrt(3) / 6, 2, lambda x: -1 / math.sqrt(3) * x - 1, lambda x: math.sqrt(3) * (1 - x))[0]

    Ne000 = 3.0 / 8 * P000_101 + 1.0 / 4 * P000_001 + 5.0 / 8 * P000_111 + 1.0 / 4 * P000_011

    P101_000 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -0.5 - math.sqrt(3) / 2, -0.5 - math.sqrt(3) / 6,
                lambda x: -math.sqrt(3) * x - math.sqrt(3) - 1, lambda x: -1 / math.sqrt(3) * x - 1 / math.sqrt(3))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -0.5 - math.sqrt(3) / 6, 0,
                lambda x: -0.5 - math.sqrt(3) / 2, lambda x: -(x + 1) / math.sqrt(3))[0]
    P101_100 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -float('inf'), -math.sqrt(3) - 1,
                lambda x: -x - 0.5 + math.sqrt(3) / 2, lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -math.sqrt(3) - 1, -math.sqrt(3) / 2 - 0.5,
                lambda x: - 1 / math.sqrt(3), lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -math.sqrt(3) / 2 - 0.5, 0,
                lambda x: 1 / math.sqrt(3) * (x + 1) + 1,lambda x: float('inf'))[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0, float('inf'),
                lambda x: x + math.sqrt(3) / 2 + 0.5, lambda x: float('inf'))[0]

    P101_010 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0, 0.5 + math.sqrt(3) / 6,
                lambda x: -math.sqrt(3) * x - math.sqrt(3) * 2 / 3 - 1, lambda x: -0.5 - math.sqrt(3) / 2)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0.5 + math.sqrt(3) / 6, 0.5 + math.sqrt(3) / 2,
                lambda x: -1.0 / math.sqrt(3) * x - math.sqrt(3) * 2 / 3 - 1, lambda x: -x * math.sqrt(3))[0]

    P101_110 = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -float('inf'), -0.5 - math.sqrt(3) / 2,
                lambda x: -float('inf'), lambda x: x - math.sqrt(3) / 2 - 0.5)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , -math.sqrt(3) / 2 - 0.5, 0,
                lambda x: -float('inf'), lambda x: 1.0 / math.sqrt(3) * x - math.sqrt(3) * 2 / 3 - 1)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , 0, math.sqrt(3) / 2 + 0.5,
                lambda x: -float('inf'), lambda x: -1.0 / math.sqrt(3) * x - math.sqrt(3) * 2 / 3 - 1)[0] + \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , math.sqrt(3) / 2 + 0.5, float('inf'),
                lambda x: -float('inf'), lambda x: -x - math.sqrt(3) / 2 - 0.5)[0]

    Ne101 = 3.0 / 8 * P101_000 + 1.0 / 4 * P101_100 + 5.0 / 8 * P101_010 + 1.0 / 4 * P101_110

    Ne = Ne000 + Ne101
    return Ne

def cal_func_16QAM(SNRdB, p1 = 1.0 / 16,p2 = 1.0 / 16, p3 = 1.0 / 16):
    sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20)

    P1011_0011 = cal_integ(sigma, 0.5, 1.5, -0.5, float('inf'))
    P1011_0111 = cal_integ(sigma, 1.5, 2.5, -0.5, float('inf'))
    P1011_1111 = cal_integ(sigma, 2.5, float('inf'), -0.5, float('inf'))

    P1011_0001 = cal_integ(sigma, 0.5, 1.5, -1.5, -0.5)
    P1011_0101 = cal_integ(sigma, 1.5, 2.5, -1.5, -0.5)
    P1011_1101 = cal_integ(sigma, 2.5, float('inf'), -1.5, -0.5)

    P1011_0100 = cal_integ(sigma, 1.5, 2.5, -2.5, -1.5)
    P1011_1100 = cal_integ(sigma, 2.5, float('inf'), -2.5, -1.5)
    P1011_1110 = cal_integ(sigma, 2.5, float('inf'), -float('inf'), -2.5)

    Ne1011 = (P1011_0011 / 8 + P1011_0111 / 4 + P1011_1111 / 8 \
                           + P1011_0001 / 8 + P1011_0101 * 3 / 8 + P1011_1101 / 4 \
                                              + P1011_0100 / 4 + P1011_1100 * 3 / 8 + P1011_1110 / 8) / 4

    P0011_1011 = cal_integ(sigma, -float('inf'), -0.5, -0.5, float('inf'))
    P0011_0111 = cal_integ(sigma, 0.5, 1.5, -0.5, float('inf'))
    P0011_1111 = cal_integ(sigma, 1.5, float('inf'), -0.5, float('inf'))
    P0011_1001 = cal_integ(sigma, -float('inf'), -0.5, -1.5, -0.5)
    P0011_0001 = cal_integ(sigma, -0.5, 0.5, -1.5, -0.5)
    P0011_0101 = cal_integ(sigma, 0.5, 1.5, -1.5, -0.5)
    P0011_1101 = cal_integ(sigma, 1.5, float('inf'), -1.5, -0.5)

    P0011_1000 = cal_integ(sigma, -float('inf'), -0.5, -2.5, -1.5)
    P0011_0000 = cal_integ(sigma, -0.5, 0.5, -2.5, -1.5)
    P0011_0100 = cal_integ(sigma, 0.5, 1.5, -2.5, -1.5)
    P0011_1100 = cal_integ(sigma, 1.5, float('inf'), -2.5, -1.5)

    P0011_1010 = cal_integ(sigma, -float('inf'), -0.5, -float('inf'), -2.5)
    P0011_0010 = cal_integ(sigma, -0.5, 0.5, -float('inf'), -2.5)
    P0011_0110 = cal_integ(sigma, 0.5, 1.5, -float('inf'), -2.5)
    P0011_1110 = cal_integ(sigma, 1.5, float('inf'), -float('inf'), -2.5)

    Ne0011 = (P0011_1011 / 16 + P0011_0111 / 16 + P0011_1111 / 8 + P0011_1001 / 8 + P0011_0001 / 16 + P0011_0101 / 8 + P0011_1101 * 3 / 16 \
             + P0011_1000 * 3 / 16 + P0011_0000 / 8 + P0011_0100 * 3 / 16 + P0011_1100 / 4 \
             + P0011_1010 / 8 + P0011_0010 / 16 + P0011_0110 / 8 + P0011_1110 * 3 / 16) / 4

    P0001_1011 = cal_integ(sigma, -float('inf'), -0.5, 0.5, float('inf'))
    P0001_0011 = cal_integ(sigma, -0.5, 0.5, 0.5, float('inf'))
    P0001_0111 = cal_integ(sigma, 0.5, 1.5, 0.5, float('inf'))
    P0001_1111 = cal_integ(sigma, 1.5, float('inf'), 0.5, float('inf'))

    P0001_1001 = cal_integ(sigma, -float('inf'), -0.5, -0.5, 0.5)
    P0001_0101 = cal_integ(sigma, 0.5, 1.5, -0.5, 0.5)
    P0001_1101 = cal_integ(sigma, 1.5, float('inf'), -0.5, 0.5)

    P0001_1000 = cal_integ(sigma, -float('inf'), -0.5, -1.5, -0.5)
    P0001_0000 = cal_integ(sigma, -0.5, 0.5, -1.5, -0.5)
    P0001_0100 = cal_integ(sigma, 0.5, 1.5, -1.5, -0.5)
    P0001_1100 = cal_integ(sigma, 1.5, float('inf'), -1.5, -0.5)

    P0001_1010 = cal_integ(sigma, -float('inf'), -0.5, -float('inf'), -1.5)
    P0001_0010 = cal_integ(sigma, -0.5, 0.5, -float('inf'), -1.5)
    P0001_0110 = cal_integ(sigma, 0.5, 1.5, -float('inf'), -1.5)
    P0001_1110 = cal_integ(sigma, 1.5, float('inf'), -float('inf'), -1.5)


    #print " this one: " + str(P0001_0000)

    Ne0001 = (P0001_1011 / 8 + P0001_0011 / 16 + P0001_0111 / 8 + P0001_1111 * 3 / 16 \
             + P0001_1001 / 16 + P0001_0101 / 16 + P0001_1101 / 8 \
             + P0001_1000 / 8 + P0001_0000 / 16 + P0001_0100 / 8 + P0001_1100 * 3 / 16 \
             + P0001_1010 * 3 / 16 + P0001_0010 / 8 + P0001_0110 * 3 / 16 + P0001_1110 / 4) / 4

    Ne = 16 * (4 * Ne1011 * p1 + 8 * Ne0011 * p2 + 4 * Ne0001 * p3)

    return Ne

def cal_func_16QAM_change(SNRdB, p1 = 1.0,p2 = 1.0, p3 = 1.0):
    return cal_func_16QAM(SNRdB, p1, p2, p3)

def cal_integ(sigma, a, b, c, d):
    integ_result, err = \
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , a, b, lambda x: c, lambda x: d)
    return integ_result

def calculation_fit(mode, lenth):
    x0 = []
    y0 = []
    for num in range(0, lenth):
        SNRdB = float(num/1.0)
        sigma = 10 ** (-SNRdB / 20)

        result = cal_func(SNRdB, mode)

        print "SNRdB is " + str(SNRdB) + ", o is " + str(sigma)
        print "sum result is " + str(result)
        x0.append(SNRdB)
        #y0.append(math.log(result, 10))
        y0.append(result)

    return  plot_display(x0, y0, 'BER - SNR',mode)

def calculation_inversion(left, right, x1, x2, f1, f2, epsilon, step, count, inversion_count, inversion_value):
    count = count + 1
    inversion_count.append(count)
    inversion_value.append((right - left))
    print str(count) + " , result is " + str((right + left) / 2) + ', error is ' + str(right - left)
    #step 1
    if step <= 1:
        x2 = left + 0.618 * (right - left)
        f2 = abs(cal_func(x2) - 0.02)

    #step 2
    if step <= 2:
        x1 = left + 0.382 * (right - left)
        f1 = abs(cal_func(x1) - 0.02)
    #step 3
    if step <= 3:
        if abs(right - left) < epsilon :
            print "ok! " +  str((right + left) / 2)
            return (right + left) / 2
    #step 4
    if step <= 4:
        if f1 < f2:
            right = x2
            x2 = x1
            f2 = f1
            return calculation_inversion(left, right, x1, x2, f1, f2, epsilon, 2, count, inversion_count, inversion_value)
        if f1 == f2:
            left = x1
            right = x2
            return calculation_inversion(left, right, x1, x2, f1, f2, epsilon, 1, count, inversion_count, inversion_value)
        if f1 > f2:
            left = x1
            x1 = x2
            f1 = f2
            #return calculation_inversion(left, right, x1, x2, f1, f2, epsilon, 5, count, inversion_count, inversion_value)

    #step 5
    if step <= 5:
        x2 = left + 0.618 * (right - left)
        f2 = abs(cal_func(x2) - 0.02)
        return calculation_inversion(left, right, x1, x2, f1, f2, epsilon, 3, count, inversion_count, inversion_value)

def shoot_target(coordinates, mode, SNRdB, point_num):
    Ps = 0
    for coor in coordinates[mode]:
        Ps += math.sqrt(coor[0]**2 + coor[1]**2)
        print math.sqrt(coor[0]**2 + coor[1]**2)
    Ps /= len(coordinates[mode])

    print Ps

    if mode == 'squre':
        sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20)
    if mode == 'star':
        sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20)
    if mode == '8QAM':
        sigma = (3 + math.sqrt(3)) / 4 * (10 ** (-SNRdB / 20))
    plt.figure('shoot_target')
    plt.title('shoot_target')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    for num in range(0, point_num):
        x_st = []
        y_st = []
        count = 0
        for coor in coordinates[mode]:
            x_st.append(coor[0])
            y_st.append(coor[1])
            count += 1
        xn = x_st + np.random.normal(loc=0.0, scale=sigma, size=len(x_st))
        yn = y_st + np.random.normal(loc=0.0, scale=sigma, size=len(x_st))
        plt.scatter(xn[:], yn[:], 2, "red")
    return

if __name__ == "__main__":
    coordinates = {}
    coordinates['squre'] = [[-1.5, 1.5],  [-0.5, 1.5],  [0.5, 1.5],  [1.5, 1.5],
                         [-1.5, 0.5],  [-0.5, 0.5],  [0.5, 0.5],  [1.5, 0.5],
                         [-1.5, -0.5], [-0.5, -0.5], [0.5, -0.5], [1.5, -0.5],
                         [-1.5, -1.5], [-0.5, -1.5], [0.5, -1.5], [1.5, -1.5], ]
    coordinates['8QAM'] = [[0, 0.5 + math.sqrt(3) / 2],
                        [-0.5, 0.5], [0.5, 0.5],
                        [-0.5 - math.sqrt(3) / 2, 0], [0.5 + math.sqrt(3) / 2, 0],
                        [-0.5, -0.5], [0.5, -0.5],
                        [0, -0.5 - math.sqrt(3) / 2], ]
    coordinates['star'] = []
    for ang in range(0, 8):
        coordinates['star'].append([math.sin(ang * math.pi / 4), math.cos(ang * math.pi / 4)])
        coordinates['star'].append([2 * math.sin(ang * math.pi / 4), 2 * math.cos(ang * math.pi / 4)])
    shoot_target(coordinates, '8QAM', 5, 100)
    ''''
    p1 = sympy.Symbol('p1')
    p2 = sympy.Symbol('p2')
    p3 = 1.0 / 16
    #print sympy.solve([4 * p1 + 8 * p2 + 4 * p3 - 1, 4 * p1 * sympy.log(p1, 2) + 8 * p2 * sympy.log(p2, 2) + 4 * p3 * math.log(p3, 2) + 3], [p1, p2])
    #print sympy.solve([4 * p1 + 8 * p2 + 4 * p3 - 1, sympy.log(p1, 2) - p3], [p1, p2])

    ax = []
    ax_line = []

    ax.append(calculation_fit('QPSK', 30))
    ax_line.append('QPSK')
    ax.append(calculation_fit('8QAM', 30))
    ax_line.append('8QAM')

    ax.append(calculation_fit('16QAM', 30))
    ax_line.append('16QAM')
    ax.append(calculation_fit('16QAM_change', 30))
    ax_line.append('16QAM_change')
    plt.legend(ax, ax_line, loc='lower left')

    inversion_count = []
    inversion_value = []
    '''
    #result = calculation_inversion(0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 1, 0, inversion_count, inversion_value)
    #plot_display(inversion_count, inversion_value, 'QPSK', 'Inversion')
    #print "calculation_inversion :" + str(result)
    #print "reslut is " + str(cal_func(result))
    plt.show()
