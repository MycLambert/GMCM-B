#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import numpy as np
from scipy.integrate import dblquad  # dblquad用于二重积分
import matplotlib.pyplot as plt
from scipy import optimize
import sympy
import random


def func(x, a, b, c, d, e):
    f = a * x * x * x * x + b * x * x * x + c * x * x + d * x + e
    return f.ravel()

def plot_display(x0, y0, figname, line):
    # 曲线拟合与绘制
    display = plt.figure(figname)
    #plt.scatter(x0[:], y0[:], 25, "red", 'x')
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
    plt.ylim(10**(-3.5), 1)
    #plt.ylim(0.1, 1)
    plt.xlim(0, 25)
    #ax1, = plt.plot(x0, y0)

    return ax
    #print popt

def cal_func(SNRdB, mode = 'QPSK', p = []):
    if mode == 'QPSK':
        return cal_func_QPSK(SNRdB)
    if mode == '8QAM':
        return cal_func_8QAM(SNRdB)
    if mode == '16QAM':
        return cal_func_16QAM(SNRdB)
    if mode == '16QAM_change':
        return cal_func_16QAM_change(SNRdB, 0.1932, 0.0284, 10 ** (-6))
    if mode == 'QPSK_target':
        return shoot_target(coordinates, 'QPSK', SNRdB, 1000)
    if mode == '8QAM_target':
        return shoot_target(coordinates, '8QAM', SNRdB, 2000)
    if mode == '16QAM_star_target':
        return shoot_target(coordinates, 'star', SNRdB, 2000, p)
    if mode == '16QAM_square_target':
        return shoot_target(coordinates, 'square', SNRdB, 2000)

def cal_func_QPSK(SNRdB):
    sigma = math.sqrt(2.0) / 2 * (10.0 ** (-SNRdB / 20.0))
    P11_10 = cal_integ(sigma, 0.5, float('inf'), -0.5, float('inf'))
    P11_00 = cal_integ(sigma, 0.5, float('inf'), -float('inf'), -0.5)
    Ne = (3 * P11_10 + 1 * P11_00) / 2
    return Ne

def cal_func_8QAM(SNRdB):
    sigma = math.sqrt((3 + math.sqrt(3)) / 4) * (10.0 ** (-SNRdB / 20.0))
    sigma = math.sqrt(((3 + math.sqrt(3)) / 4) * (1.0 / (10.0 ** (SNRdB / 10.0))))
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
    sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20.0)

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
        dblquad(lambda x, y: (1.0 / (2.0 * math.pi * sigma ** 2.0)) * math.exp(
            -1.0 / 2.0 * ((x / sigma) ** 2.0 + (y / sigma) ** 2.0))
                , a, b, lambda x: c, lambda x: d)
    return integ_result

def calculation_fit(mode, lenth, figure, p = []):
    x0 = []
    y0 = []
    for num in range(0, lenth):
        SNRdB = float(num/1.0)
        sigma = 10 ** (-SNRdB / 20)

        result = cal_func(SNRdB, mode, p)

        print "SNRdB is " + str(SNRdB) + ", o is " + str(sigma)
        print "sum result is " + str(result)
        x0.append(SNRdB)
        #y0.append(math.log(result, 10))
        y0.append(result)

    return  plot_display(x0, y0, figure, mode)

def calculation_inversion(left, right, x1, x2, f1, f2, epsilon, step, count, inversion_count, inversion_value):
    count = count + 1
    inversion_count.append(count)
    inversion_value.append((right - left))
    #print str(count) + " , result is " + str((right + left) / 2) + ', error is ' + str(right - left)
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

def shoot_target(coordinates, mode, SNRdB, point_num, p = [2,1,1,1]):
    err_num = 0
    Ps = 0
    for coor in coordinates[mode]:
        Ps += (coor[0]**2 + coor[1]**2)
        #print (coor[0]**2 + coor[1]**2)
    Ps /= len(coordinates[mode])
    print math.sqrt(Ps)

    sigma = math.sqrt(Ps) * 10 ** (-SNRdB / 20.0)
    print sigma

    if mode == 'square':
        sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20.0)
    if mode == 'star':
        sigma = math.sqrt(2.5) * 10 ** (-SNRdB / 20.0)
    if mode == '8QAM':
        sigma = math.sqrt((3 + math.sqrt(3)) / 4) * (10 ** (-SNRdB / 20.0))

    plt.figure(u'16QAM仿真')
    plt.title(u'16QAM_矩形_仿真')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    p += [2,1,1,1,1,]

    p1_1 = p[0]/2
    p1_2 = p1_1
    p2 = p[1]
    p3 = p[2]
    p_sum = 2 * p1_1 + p2 + p3
    for num0 in range(0, point_num):
    #for num0 in range(5, 6):

        #print distance([xn[num], yn[num]], [0, 0])
        num = 0
        x_st = []
        y_st = []
        x_err = []
        y_err = []
        count = 0
        num_st = []

        for coor in coordinates[mode]:
            if mode == '8QAM':
                ran_base = random.randint(0, 7)
                x_st.append(coordinates[mode][ran_base][0])
                y_st.append(coordinates[mode][ran_base][1])
                num_st.append(ran_base)
            else:
                ran_num = random.random() * p_sum
                ran_base = random.randint(0, 3)
                if ran_num <= p1_1:
                    num = 4 * ran_base
                    # print "1_1"
                elif ran_num <= p1_1 + p1_2:
                    num = 4 * ran_base + 1
                    # print "1_2"
                elif ran_num <= p1_1 + p1_2 + p2:
                    num = 4 * ran_base + 2
                    # print "2"
                else:
                    num = 4 * ran_base + 3

                x_st.append(coordinates[mode][num][0])
                y_st.append(coordinates[mode][num][1])
                num_st.append(num)
            count += 1


        cov = [[sigma**2, 0], [0, sigma**2]]
        #array = mean + np.random.multivariate_normal([0, 0], cov, len(x_st))#xn, yn
        mu = np.array([[0, 0]])
        Sigma = np.array(cov)
        R = np.linalg.cholesky(Sigma)
        s = np.dot(np.random.randn(len(x_st), 2), R) + mu
        xn = []
        yn = []
        for posi in s:
            #print posi
            xn.append(posi[0])
            yn.append(posi[1])
        xn = list_add(x_st, xn)
        yn = list_add(y_st, yn)
        #yn = y_st + np.random.normal(loc=0.0, scale=sigma, size=len(x_st))
        for uu_num in range(0, len(coordinates[mode])):
            dis_should = distance([xn[uu_num], yn[uu_num]], [coordinates[mode][num_st[uu_num]][0], coordinates[mode][num_st[uu_num]][1]])
            #print dis_should
            flag = 0
            char_diff = 0
            err_NN = 0
            for num_real in range(0, len(coordinates[mode])):
                dis_real = distance([xn[uu_num], yn[uu_num]], [coordinates[mode][num_real][0], coordinates[mode][num_real][1]])
                if dis_real < dis_should:
                    flag = 1
                    dis_should = dis_real
                    err_NN = num_real
                    #print str(num) + " - " + str(num_real) + " !!!!!! " + str(dis_real)
                    #print str(coordinates[mode][num][2]) + " - " + str(coordinates[mode][num_real][2])
                    #print str(char_diff) + " : " + str(coordinates[mode][num_real][2]) + " - " + coordinates[mode][num][2]
                    #print str(dis_real) + " - " + str(dis_should)
                #print "num:" + str(xn[num])
            if flag == 1:
                x_err.append(xn[uu_num])
                y_err.append(yn[uu_num])
                for num_char in range(0, len(coordinates[mode][err_NN][2])):
                    if coordinates[mode][err_NN][2][num_char] != coordinates[mode][num][2][num_char]:
                        char_diff += 1.0
                err_num += char_diff

        plt.scatter(xn[:], yn[:], 10, "blue", 'o')
        plt.scatter(x_err[:], y_err[:], 10, "red", 'o')
    BER = err_num / len(coordinates[mode][num_real][2]) / point_num / len(coordinates[mode])
    print "SNRdB is : " + str(SNRdB) + ", BER is : " + str(BER)
    return BER
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def star_c(lenth, ang, code, p):
    result = [lenth * math.sin(ang * math.pi / 4), lenth * math.cos(ang * math.pi / 4), code]
    #print result
    return result

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

if __name__ == "__main__":
    coordinates = {}
    coordinates['QPSK'] = [[-0.5, 0.5, '11'], [0.5, 0.5, '10'], [-0.5, -0.5, '00'], [0.5, -0.5, '01'], ]
    coordinates['square'] = [[-1.5, 1.5, '1011'],  [-0.5, 1.5, '0011'],  [0.5, 1.5, '0111'],  [1.5, 1.5, '1111'],
                         [-1.5, 0.5, '1001'],  [-0.5, 0.5, '0001'],  [0.5, 0.5, '0101'],  [1.5, 0.5, '1101'],
                         [-1.5, -0.5, '1000'], [-0.5, -0.5, '0000'], [0.5, -0.5, '0100'], [1.5, -0.5, '1100'],
                         [-1.5, -1.5, '1010'], [-0.5, -1.5, '0010'], [0.5, -1.5, '0110'], [1.5, -1.5, '1110'], ]
    coordinates['8QAM'] = [[0, 0.5 + math.sqrt(3) / 2, '101'],
                        [-0.5, 0.5, '000'], [0.5, 0.5, '001'],
                        [-0.5 - math.sqrt(3) / 2, 0, '100'], [0.5 + math.sqrt(3) / 2, 0, '111'],
                        [-0.5, -0.5, '010'], [0.5, -0.5, '011'],
                        [0, -0.5 - math.sqrt(3) / 2, '110'], ]
    coordinates['star'] = [star_c(1, 0, '0000', 0.0001), star_c(1, 1, '0010', 0.0001), star_c(2, 0, '0001', 0.1107), star_c(2, 1, '0011', 0.1391),
                           star_c(1, 2, '0110', 0.0001), star_c(1, 3, '0100', 0.0001), star_c(2, 2, '0111', 0.1107), star_c(2, 3, '0101', 0.1391),
                           star_c(1, 4, '1100', 0.0001), star_c(1, 5, '1110', 0.0001), star_c(2, 4, '1101', 0.1107), star_c(2, 5, '1111', 0.1391),
                           star_c(1, 6, '1010', 0.0001), star_c(1, 7, '1000', 0.0001), star_c(2, 6, '1011', 0.1107), star_c(2, 7, '1001', 0.1391),]
    #shoot_target(coordinates, 'QPSK', 9.9093152729, 100)

    p1 = sympy.Symbol('p1')
    p2 = sympy.Symbol('p2')
    p3 = 1.0 / 16
    #print sympy.solve([4 * p1 + 8 * p2 + 4 * p3 - 1, 4 * p1 * sympy.log(p1, 2) + 8 * p2 * sympy.log(p2, 2) + 4 * p3 * math.log(p3, 2) + 3], [p1, p2])
    #print sympy.solve([4 * p1 + 8 * p2 + 4 * p3 - 1, sympy.log(p1, 2) - p3], [p1, p2])

    ax = []
    ax_line = []

    #ax.append(calculation_fit('QPSK', 25,'BER - SNR'))
    #ax_line.append('QPSK')
    #ax.append(calculation_fit('8QAM', 25,'BER - SNR'))
    #ax_line.append('8QAM')
    #ax.append(calculation_fit('16QAM', 25,'BER - SNR'))
    #ax_line.append('16QAM')
    #ax.append(calculation_fit('16QAM_change', 30,'BER - SNR'))
    #ax_line.append('16QAM_change')
    #ax.append(calculation_fit('QPSK_target', 25, 'BER - SNR'))
    #ax_line.append('QPSK_target')
    #ax.append(calculation_fit('8QAM_target', 25, 'BER - SNR'))
    #ax_line.append('8QAM_target')
    #ax.append(calculation_fit('16QAM_square_target', 25, 'BER - SNR'))
    #ax_line.append('16QAM_square_target')
    ax.append(calculation_fit('8QAM_target', 8, 'BER - SNR'))
    ax_line.append(u'8QAM_仿真')
    ax.append(calculation_fit('16QAM_star_target', 8, 'BER - SNR', [0.1, 0.1107, 0.139]))
    ax_line.append(u'16QAM_星型_仿真')
    #ax.append(calculation_fit('16QAM_star_target', 25, 'BER - SNR', []))
    #ax_line.append('16QAM_star_target_all=1')
    ax.append(calculation_fit('16QAM_square_target', 8, 'BER - SNR'))
    ax_line.append(u'16QAM_矩形_仿真')
    plt.legend(ax, ax_line, loc='lower left', )

    ''''
    inversion_count = []
    inversion_value = []

    result = calculation_inversion(0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 1, 0, inversion_count, inversion_value)
    plot_display(inversion_count, inversion_value, '8QAM', 'Inversion')
    print "calculation_inversion :" + str(result)
    print "QPSK reslut is " + str(cal_func(result))
    '''
    STR = '0101'
    for char in STR:
        print char
    plt.show()
