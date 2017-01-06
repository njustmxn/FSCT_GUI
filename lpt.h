/*
 * lpt.h与lpt.c实现了普通图像的对数极坐标变换, 该变换过程首先根
 * 据源图像尺寸与目标图像尺寸建立一个LPT_Grid结构, 该结构中定义
 * 了目标图像每个像素在源图像中映射的坐标, 然后通过使用双线性插值
 * 的方法, 实现图像的变换.
 * 其中, 双线性插值使用了浮点运算转整形的优化技术, 仅需在建立插值
 * 网格时需要浮点运算, 在后续的变换过程中全程使用整形数的加减乘与
 * 移位运算, 可以大量的节省计算时间. 尤其适用于对连续图像序列进行
 * 统一参数的对数极坐标变换.
 * 此外, 该源码实现的对数极坐标变换引入的"最小极径系数"的概念, 避
 * 免在变换中心产生大面积渐变模式的图像, 使变换后的图像包含更多的
 * 有效信息. 最小极径系数的原理请参见作者的相关论文.
 *
 * 参考文献:
 * [1] 马晓楠, 刘晓利, 李银伢. 自适应尺度的快速相关滤波跟踪算法[J].
 *     计算机辅助设计与图形学学报, 2017. 29(?): ?-?
 *
 * njustmxn@163.com 南京理工大学
 * Created by 马小南 on 2016.8.23
 * Copyright (c) 2016 马小南. All rights reserved
 */

#ifndef LPT_H
#define LPT_H

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#define LPT_PI 3.14159265
#define LPT_EPSILON 1.19209290e-7f

typedef struct LPT_Grid
{
    int theta;
    int rho;
    int imgWidth;
    int imgHeight;
    float rhoMinRate;
    int *xGrid;
    int *yGrid;
} LPT_Grid;

LPT_Grid* newLptGrid(int imgWidth, int imgHeight, int gridWidth, int gridHeight, float rhoMinRate);

void freeLptGrid(LPT_Grid *lpt);

void logPolar(const unsigned char *src, unsigned char *dst, LPT_Grid *lpt);

#ifdef __cplusplus
}
#endif //__cplusplus

#endif // LPT_H
