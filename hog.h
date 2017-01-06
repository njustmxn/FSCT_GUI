/*
 * hog.h与hog.c 实现了HOG(Histogram of Oriented Gradients)特征的提取工作,
 * 采用的方法为UocTTi版本(2010年), 这种HOG特征比传统的DalalTriggs版本(2005年)
 * 在物体识别上具有更高的性能, 而且特征的维数较后者少, 在以Correlation Tracking
 * 为基础的目标跟踪算法中得到广泛应用. 源码的实现过程同时参考了OpenCV库中的DalalTriggs
 * 方法以及VLFeat中的UocTTi方法, 结合二者各自的有点, 实现一个更高效, 更便捷的版本.
 * 并且脱离了OpenCV与VLFeat较大的框架, 去除各种相互依赖的头文件, 自成一体, 直接使
 * 用C的标准库实现, 方便跨平台使用.
 *
 * 参考文献:
 * [1] P. F. Felzenszwalb, R. B. Grishick, D. McAllester, and D. Ramanan.
 *     Object detection with discriminatively trained part based models.
 *     PAMI, 2010.
 * [2] N. Dalal and B. Triggs. Histograms of oriented gradients for human
 *     detection. In Proc. CVPR, 2005.
 * 参考源码库:
 * 1. http://www.vlfeat.org/index.html
 * 2. http://opencv.org/
 *
 * njustmxn@163.com 南京理工大学
 * Created by 马小南 on 2016.8.20
 * Copyright (c) 2016 马小南. All rights reserved
 *
 */


#ifndef _HOG_H_
#define _HOG_H_

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#define HOG_PI 3.14159265
#define HOG_EPSILON 1.19209290e-7

#define MIN_VAL(x,y) (((x)<(y))?(x):(y))
#define MAX_VAL(x,y) (((x)>(y))?(x):(y))

typedef struct FHOG
{
    int dimension;          //HOG特征的维数
    int nOrientation;       //梯度方向数
    int cellSize;           //单个cell的尺寸(正方形)
    int histWidth;          //梯度直方图宽度(cell横向个数)
    int histHeight;         //梯度直方图高度(cell纵向个数)
    float *hist;            //有方向梯度直方图buffer
    float *histNorm;        //梯度直方图归一化buffer
    int glyphSize;          //单个cell的可视化HOG特征尺寸
    unsigned char *glyphs;  //HOG特征可视化缓存
    float *gammaLut;        //伽马校正查找表
} FHOG;

FHOG* newHogDescriptor(int cellsize, int numOrientations, int glyph, int gammaCorrection);

void freeHogDescriptor(FHOG* self);

int getHogFeatureSize(const FHOG* self, int width, int height);

int getHogFeatureCols(const FHOG* self, int width);

int getHogFeatureRows(const FHOG* self, int height);

int getHogFeatureChannels(const FHOG* self);

int getHogRenderWidth(const FHOG* self, int width);

int getHogRenderHeight(const FHOG* self, int height);


void calcHogFeature(FHOG* self, const unsigned char *image, int width, int height, float* features);

int getHogFeatureGlyphSize(const FHOG* self);

void renderHogFeature(const FHOG* self, const float* features, unsigned char* vImg);


#ifdef __cplusplus
}
#endif //__cplusplus

#endif //_HOG_H_
