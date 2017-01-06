#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "hog.h"

static const float divSqrt18 = 1.0/4.2426; // 1/sqrt(18)
static float fastAtan2(float y, float x);
static void prepareBuffers(FHOG* self, int width, int height);
static void normalizeHist(FHOG* self, float* features);


/**
 * @brief 使用指定参数创建一个HOG描述子
 * @param cellsize cell单元尺寸(cell为正方形)
 * @param nOrientations 梯度方向数
 * @param noGlyph 是否需要HOG特征可视化
 * @param gammaCorrection 是否需要图像伽马校正
 * @return HOG描述子结构
 */
FHOG *newHogDescriptor(int cellsize, int nOrientation, int glyph, int gammaCorrection)
{
    int i, j, k;
    FHOG *self = (FHOG*)malloc(sizeof(FHOG));
    assert(self != NULL);
    assert(nOrientation >= 1 && cellsize >= 1);
    self->nOrientation = nOrientation;
    self->cellSize = cellsize;
    /* 假如nOrientation=9, 则有18个有方向梯度与9个无方向梯度, 以及4层纹理特征 */
    self->dimension = 3 * nOrientation + 4;
    self->histWidth = 0;
    self->histHeight = 0;
    self->hist = NULL;
    self->histNorm = NULL;
    /* 根据是否需要伽马校正, 决定是否需要建立查找表并初始化 */
    if (!gammaCorrection)
        self->gammaLut = NULL;
    else
    {
        self->gammaLut = (float*)malloc(sizeof(float) * 256);
        for (i = 0; i < 256; i++)
            self->gammaLut[i] = (float)sqrt((float)i);
    }
    self->glyphSize = 21;
    /* 根据是否需要进行HOG特征可视化, 决定是否需要创建可视化查找表 */
    if(!glyph)
        self->glyphs = NULL;
    else
    {
        /* 创建梯度特征可视化图像查找表, 每个cell作为一个可视化单元, 该单元尺寸
         * 为长和宽都为self->glyphSize, 共有nOrientation层
         */
        self->glyphs = (unsigned char*)malloc(self->glyphSize * self->glyphSize * nOrientation * sizeof(unsigned char));
        memset(self->glyphs, 0, sizeof(unsigned char) * self->glyphSize * self->glyphSize * nOrientation);

#define atglyph(x,y,k) self->glyphs[(x) + self->glyphSize * (y) + self->glyphSize * self->glyphSize * (k)]

        for (k = 0; k < self->nOrientation; k++)
        {
            float angle = fmod(k * HOG_PI / nOrientation + HOG_PI * 0.5, HOG_PI) ;
            float x2 = self->glyphSize * cos(angle) * 0.5 ;
            float y2 = self->glyphSize * sin(angle) * 0.5 ;

            if (angle <= HOG_PI / 4 || angle >= HOG_PI * 3 / 4)
            {
                /* along horizontal direction */
                float slope = y2 / x2 ;
                float offset = (1 - slope) * (self->glyphSize - 1) / 2 ;
                int skip = (1 - fabs(cos(angle))) * self->glyphSize / 2 ;
                for (i = skip; i < self->glyphSize - skip; i++)
                {
                    j = floor(slope * i + offset + 0.5);
                    atglyph(i,j,k) = 1 ;
                }
            }
            else
            {
                /* along vertical direction */
                float slope = x2 / y2 ;
                float offset = (1 - slope) * (self->glyphSize - 1) / 2 ;
                int skip = (1 - sin(angle)) * self->glyphSize / 2 ;
                for (j = skip; j < self->glyphSize - skip; j++)
                {
                    i = floor(slope * j + offset + 0.5) ;
                    atglyph(i,j,k) = 1 ;
                }
            }
        }
    }
#undef atglyph
    return self;
}


/**
 * @brief 释放一个HOG描述子结构
 * @param self
 */
void freeHogDescriptor(FHOG* self)
{
    if (self->glyphs)
    {
        free(self->glyphs);
        self->glyphs = NULL;
    }
    if (self->hist)
    {
        free(self->hist);
        self->hist = NULL;
    }
    if (self->histNorm)
    {
        free(self->histNorm);
        self->histNorm = NULL;
    }
    if (self->gammaLut)
    {
        free(self->gammaLut);
        self->gammaLut = NULL;
    }
    free(self);
    return;
}

/**
 * @brief 获取HOG特征的尺寸(此函数须先于calcHogFeature()调用, 以提前为feature分配空间)
 * @param self HOG描述子
 * @param width 待提取HOG特征的图像的宽度
 * @param height 待提取HOG特征的图像的高度
 * @return
 */
int getHogFeatureSize(const FHOG *self, int width, int height)
{
    int histWidth = (width + self->cellSize / 2) / self->cellSize;
    int histHeight = (height + self->cellSize / 2) / self->cellSize;
    return (histWidth * histHeight * self->dimension);
}

/**
 * @brief 计算输入图像的HOG特征
 * @param self HOG描述子
 * @param image 源图像
 * @param width 宽度
 * @param height 高度
 * @param features HOG特征向量(须预先分配尺寸合适的内存空间)
 */
void calcHogFeature(FHOG* self, const unsigned char* image, int width, int height, float* features)
{
    int nbins = self->nOrientation * 2; //2*PI对应的方向个数
    float angleScale = (float)(self->nOrientation / HOG_PI);
    int x, y, k;
    int histStride, histLayerStride;
    prepareBuffers(self, width, height);
    histStride = self->histWidth * self->histHeight;
    histLayerStride = histStride * self->nOrientation;
    assert(self != NULL && image != NULL && self->hist != NULL
           && self->histNorm != NULL && features != NULL);

#define at(x,y,k) (self->hist[(x) + (y) * self->histWidth + (k) * histStride])

    for (y = 1; y < height-1; y++)
    {
        /* 源图像的行指针 */
        const unsigned char* curPtr = image + width * y;
        const unsigned char* prevPtr = image + width * (y - 1);
        const unsigned char* nextPtr = image + width * (y + 1);
        for (x = 1; x < width-1; x++)
        {
            /* +--------+--------+
             * |        |        |
             * | (1, 1) | (2, 1) |
             * |        |        |
             * +--------+--------+
             * |        |        |
             * | (1, 2) | (2, 2) |
             * |        |        |
             * +--------+--------+
             * 如上图所示, 每一块代表一个cell, cell(1,1)中的的像素坐标总是能同时辐射到其
             * 右方,下方和右下方的3个cell, 此时要根据实际距离进行权重的线性插值运算, 把当前
             * 坐标的梯度分配给周围4个cell.
             * 特殊情况:
             * 1. 当cell(1,1)为图像最右方cell时, 仅能辐射到cell(1,2);
             * 2. 当cell(1,1)为图像最下方cell时, 仅能辐射到cell(2,1);
             * 3. 当cell(1,1)为图像最右下方cell时, 无cell与其构成共享关系;
             */
            float cellX = x / self->cellSize; //x方向cell归属值
            float cellY = y / self->cellSize; //y方向cell归属值
            int cellIdx = (int)floor(cellX); //x方向截断的cell标号
            int cellIdy = (int)floor(cellY); //y方向截断的cell标号
            float wx2 = cellX - cellIdx; //x方向上分配给2号cell的权重
            float wy2 = cellY - cellIdy; //y方向上分配给2号cell的权重
            float wx1 = 1.0 - wx2; //x方向上分配给1号cell的权重
            float wy1 = 1.0 - wy2; //y方向上分配给1号cell的权重

            /* 梯度及其所属方向的计算 */
            int idx;
            float gradWeight[2] = {0.0, 0.0}; //相邻两个方向经线性插值后的梯度值
            int binId[2] = {-1, -1}; //相邻两个方向的索引号
            float dx, dy, mag, angle;
            if(self->gammaLut)
            {
                dx = self->gammaLut[curPtr[x+1]] - self->gammaLut[curPtr[x-1]]; //x方向梯度
                dy = self->gammaLut[nextPtr[x]] - self->gammaLut[prevPtr[x]]; //y方向梯度
            }
            else
            {
                dx = curPtr[x+1] - curPtr[x-1]; //x方向梯度
                dy = nextPtr[x] - prevPtr[x]; //y方向梯度
            }
            mag = (float)sqrt(dx * dx + dy * dy); //梯度幅值
            angle = fastAtan2(dy, dx); //梯度的角度
            /* 保存该梯度方向在左右相邻的bin的模, 模值分配采用线性插值 */
            angle = angle * angleScale; //每一格角度为pi/9, t落在第t/(pi/9)格
            idx = (int)floor(angle);
            angle -= idx;
            gradWeight[0] = mag * (1.f - angle);
            gradWeight[1] = mag * angle;
            if (idx < 0)
                idx += nbins;
            else if (idx >= nbins)
                idx -= nbins;
            assert(idx < nbins);
            /* 保存与该梯度方向相邻的左右两个bin编号 */
            binId[0] = idx;
            idx++;
            /* 如果idx < nbins, idx还是它自身; 如果超过了, 就算bin[0] */
            binId[1] = (idx < nbins ? idx : 0);
            /* 将梯度在两个方向上的加权值分配到与其相邻的cell中 */
            if(cellIdx < self->histWidth - 1 && cellIdy < self->histHeight - 1)
            {
                /* 常规情况, 辐射4个cell */
                at(cellIdx, cellIdy, binId[0]) += gradWeight[0] * wx1 * wy1;
                at(cellIdx, cellIdy, binId[1]) += gradWeight[1] * wx1 * wy1;
                at(cellIdx + 1, cellIdy, binId[0]) += gradWeight[0] * wx2 * wy1;
                at(cellIdx + 1, cellIdy, binId[1]) += gradWeight[1] * wx2 * wy1;
                at(cellIdx, cellIdy + 1, binId[0]) += gradWeight[0] * wx1 * wy2;
                at(cellIdx, cellIdy + 1, binId[1]) += gradWeight[1] * wx1 * wy2;
                at(cellIdx + 1, cellIdy + 1, binId[0]) += gradWeight[0] * wx2 * wy2;
                at(cellIdx + 1, cellIdy + 1, binId[1]) += gradWeight[1] * wx2 * wy2;
            }
            else if(cellIdx < self->histWidth - 1)
            {
                /* 特殊情况2, 辐射x方向上2个cell */
                at(cellIdx, cellIdy, binId[0]) += gradWeight[0] * wx1;
                at(cellIdx, cellIdy, binId[1]) += gradWeight[1] * wx1;
                at(cellIdx + 1, cellIdy, binId[0]) += gradWeight[0] * wx2;
                at(cellIdx + 1, cellIdy, binId[1]) += gradWeight[1] * wx2;
            }
            else if(cellIdy < self->histHeight - 1)
            {
                /* 特殊情况1, 辐射y方向上2个cell */
                at(cellIdx, cellIdy, binId[0]) += gradWeight[0] * wy1;
                at(cellIdx, cellIdy, binId[1]) += gradWeight[1] * wy1;
                at(cellIdx, cellIdy + 1, binId[0]) += gradWeight[0] * wy2;
                at(cellIdx, cellIdy + 1, binId[1]) += gradWeight[1] * wy2;
            }
            else
            {
                /* 特殊情况3, 仅能自己独享 */
                at(cellIdx, cellIdy, binId[0]) += gradWeight[0];
                at(cellIdx, cellIdy, binId[1]) += gradWeight[1];
            }
        } //next x
    } //next y
#undef at
    /* 计算无方向梯度直方图的L2范数 */
    for(k = 0; k < self->nOrientation; k++)
    {
        float *histNormPtr = self->histNorm;
        const float *histPtr = self->hist + k * histStride;
        for(x = 0; x < histStride; x++)
        {
            float h1 = *histPtr;
            float h2 = *(histPtr + histLayerStride);
            float h = h1 + h2;
            *histNormPtr += h * h;
            histNormPtr++;
            histPtr++;
        }
    }
    normalizeHist(self, features);
    return;
}

/**
 * @brief 获取HOG特征可视化图像的尺寸
 * @param self
 * @return
 */
int getHogFeatureGlyphSize(const FHOG *self)
{
    return (self->glyphSize * self->glyphSize * self->histWidth * self->histHeight);
}

/**
 * @brief 画出HOG特征的可视化图像, 需要提前为该图像分配空间
 * @param self HOG描述子
 * @param features HOG特征
 * @param image HOG特征可视化图像
 */
void renderHogFeature(const FHOG* self, const float* features, unsigned char* vImg)
{
    int x, y, k, cx, cy;
    int histWidth = self->histWidth;
    int histHeight = self->histHeight;
    int histStride = histWidth * histHeight;
    int width = histWidth * self->glyphSize;
    int height = histHeight * self->glyphSize;
    const float *featPtr = features;
    float *fImg = (float*)malloc(sizeof(float) * width * height);
    float *maxValue = (float*)malloc(sizeof(float) * histWidth * histHeight);
    float *minValue = (float*)malloc(sizeof(float) * histWidth * histHeight);
    memset(fImg, 0, sizeof(float) * width * height);
    memset(maxValue, 0, sizeof(float) * histWidth * histHeight);
    memset(minValue, 0, sizeof(float) * histWidth * histHeight);
    assert(fImg != NULL);
    assert(self != NULL && self->glyphs != NULL);
    assert(vImg != NULL && features != NULL);
    assert(histWidth > 0 && histHeight > 0);

    for (y = 0; y < histHeight; y++)
    {
        for (x = 0; x < histWidth; x++)
        {
            float minWeight = 0;
            float maxWeight = 0;
            const unsigned char *glyph;
            float *glyphImage;
            for (k = 0; k < self->nOrientation; k++)
            {
                float weight = featPtr[k * histStride] +
                               featPtr[(k + self->nOrientation) * histStride] +
                               featPtr[(k + 2 * self->nOrientation) * histStride];
                glyph = self->glyphs + k * (self->glyphSize * self->glyphSize);
                glyphImage = fImg + self->glyphSize * x + y * histWidth *
                             self->glyphSize * self->glyphSize;
                maxWeight = MAX_VAL(weight, maxWeight);
                minWeight = MIN_VAL(weight, minWeight);
                for (cy = 0; cy < self->glyphSize; cy++)
                {
                    for (cx = 0; cx < self->glyphSize; cx++)
                    {
                        *glyphImage += weight * (*glyph);
                        glyph++;
                        glyphImage++;
                    }
                    glyphImage += (histWidth - 1) * self->glyphSize;
                }
            }
            glyphImage = fImg + self->glyphSize * x + y * histWidth *
                         self->glyphSize * self->glyphSize;
            for (cy = 0; cy < self->glyphSize; cy++)
            {
                for (cx = 0; cx < self->glyphSize; cx++)
                {
                    float value = *glyphImage;
                    *glyphImage = MAX_VAL(minWeight, MIN_VAL(maxWeight, value));
                    glyphImage++;
                }
                glyphImage += (histWidth - 1) * self->glyphSize;
            }
            minValue[x + y * histWidth] = minWeight;
            maxValue[x + y * histWidth] = maxWeight;
            featPtr++;
        }
    }
    for (y = 0; y < histHeight; y++)
    {
        for (x = 0; x < histWidth; x++)
        {
            float *glyphImage = fImg + self->glyphSize * x + y * histWidth *
                                self->glyphSize * self->glyphSize;
            unsigned char *pImg = vImg + self->glyphSize * x + y * histWidth *
                                  self->glyphSize * self->glyphSize;
            for (cy = 0; cy < self->glyphSize; cy++)
            {
                for (cx = 0; cx < self->glyphSize; cx++)
                {
                    *pImg = (*glyphImage - minValue[x + y * histWidth]) * 255 /
                            (maxValue[x + y * histWidth] - minValue[x + y * histWidth]);
                    pImg++;
                    glyphImage++;
                }
                glyphImage += (histWidth - 1) * self->glyphSize;
                pImg += (histWidth - 1) * self->glyphSize;
            }
        }
    }
    free(fImg);
    free(maxValue);
    free(minValue);
    return;
}

/**
 * @brief 分配计算HOG特征所需的缓存空间
 * @param self
 * @param width
 * @param height
 * @param cellSize
 */
static void prepareBuffers(FHOG* self, int width, int height)
{
    int histWidth = (width + self->cellSize * 0.5) / self->cellSize;
    int histHeight = (height + self->cellSize * 0.5) / self->cellSize;
    assert(width > 3 && height > 3);
    assert(histWidth > 0 && histHeight > 0);
    if(self->hist && self->histNorm &&
            self->histWidth == histWidth &&
            self->histHeight == histHeight)
    {
        /* 已经分配了尺寸合适的缓存空间, 仅需对其进行重新初始化 */
        memset(self->hist, 0, sizeof(float) * histWidth * histHeight * self->nOrientation * 2);
        memset(self->histNorm, 0, sizeof(float) * histWidth * histHeight);
    }
    else
    {
        /* 否则可能是没有分配缓存空间, 或者已经分配, 但是尺寸不一致, 需要先释放掉再重新分配 */
        if(self->hist)
        {
            free(self->hist);
            self->hist = NULL;
        }
        if(self->histNorm)
        {
            free(self->histNorm);
            self->histNorm = NULL;
        }
        /* 新分配的空间会自动初始化为零 */
        self->hist = (float*)malloc(sizeof(float) * histWidth * histHeight * self->nOrientation * 2);
        self->histNorm = (float*)malloc(sizeof(float) * histWidth * histHeight);
        memset(self->hist, 0, sizeof(float) * histWidth * histHeight * self->nOrientation * 2);
        memset(self->histNorm, 0, sizeof(float) * histWidth * histHeight);
        self->histWidth = histWidth;
        self->histHeight = histHeight;
    }
    return;
}

/**
 * @brief 归一化梯度直方图, 并生成最终的HOG特征
 * @param self
 * @param features
 */
static void normalizeHist(FHOG* self, float* features)
{
    int x, y, k;
    int histStride = self->histWidth * self->histHeight;
    const float *histPtr;
    assert(features != NULL && self !=NULL && self->hist != NULL && self->histNorm != NULL);
    /* 如下图所示, 以5为中心的cell同时属于1245, 2356, 4578, 5689这4个block,
     * 梯度方向直方图的归一化是面向block而言的, 这就需要求得每个block的L2范数,
     * 取其平方根的倒数作为该block内直方图的归一化因子, 具体到每个cell的feature,
     * 取该block内所有cell的直方图的均值即可.
     * +---+---+---+
     * | 1 | 2 | 3 |
     * +---+---+---+
     * | 4 | 5 | 6 |
     * +---+---+---+
     * | 7 | 8 | 9 |
     * +---+---+---+
     */

#define at(x,y,k) (self->hist[(x) + (y) * self->histWidth + (k) * histStride])
#define atNorm(x,y) (self->histNorm[(x) + (y) * self->histWidth])

    histPtr = self->hist;
    for (y = 0; y < self->histHeight; y++)
    {
        for (x = 0; x < self->histWidth; x++)
        {
            /* 计算当前cell索引所能共享的其他cell索引 */
            int cellIdxM = MAX_VAL(x - 1, 0);
            int cellIdxP = MIN_VAL(x + 1, self->histWidth - 1);
            int cellIdyM = MAX_VAL(y - 1, 0);
            int cellIdyP = MIN_VAL(y + 1, self->histHeight - 1);
            /* 当前cell及其周围8个cell的L2范数 */
            float norm1 = atNorm(cellIdxM, cellIdyM);
            float norm2 = atNorm(x, cellIdyM);
            float norm3 = atNorm(cellIdxP, cellIdyM);
            float norm4 = atNorm(cellIdxM, y);
            float norm5 = atNorm(x, y);
            float norm6 = atNorm(cellIdxP, y);
            float norm7 = atNorm(cellIdxM, cellIdyP);
            float norm8 = atNorm(x, cellIdyP);
            float norm9 = atNorm(cellIdxP, cellIdyP);

            float factor1, factor2, factor3, factor4;
            float t1 = 0, t2 = 0, t3 = 0, t4 = 0;
            float *featPtr = features + x + self->histWidth * y;
            /* 每个归一化因子对应该cell所属的4个block中的一个 */
            factor1 = 1.0 / sqrt(norm1 + norm2 + norm4 + norm5 + 1e-4);
            factor2 = 1.0 / sqrt(norm2 + norm3 + norm5 + norm6 + 1e-4);
            factor3 = 1.0 / sqrt(norm4 + norm5 + norm7 + norm8 + 1e-4);
            factor4 = 1.0 / sqrt(norm5 + norm6 + norm8 + norm9 + 1e-4);
            for (k = 0; k < self->nOrientation; k++)
            {
                float ha = histPtr[histStride * k];
                float hb = histPtr[histStride * (k + self->nOrientation)];
                float hc;

                float ha1 = factor1 * ha;
                float ha2 = factor2 * ha;
                float ha3 = factor3 * ha;
                float ha4 = factor4 * ha;

                float hb1 = factor1 * hb;
                float hb2 = factor2 * hb;
                float hb3 = factor3 * hb;
                float hb4 = factor4 * hb;

                float hc1 = ha1 + hb1;
                float hc2 = ha2 + hb2;
                float hc3 = ha3 + hb3;
                float hc4 = ha4 + hb4;

                ha1 = MIN_VAL(0.2, ha1);
                ha2 = MIN_VAL(0.2, ha2);
                ha3 = MIN_VAL(0.2, ha3);
                ha4 = MIN_VAL(0.2, ha4);

                hb1 = MIN_VAL(0.2, hb1);
                hb2 = MIN_VAL(0.2, hb2);
                hb3 = MIN_VAL(0.2, hb3);
                hb4 = MIN_VAL(0.2, hb4);

                hc1 = MIN_VAL(0.2, hc1);
                hc2 = MIN_VAL(0.2, hc2);
                hc3 = MIN_VAL(0.2, hc3);
                hc4 = MIN_VAL(0.2, hc4);

                t1 += hc1;
                t2 += hc2;
                t3 += hc3;
                t4 += hc4;

                ha = 0.25 * (ha1 + ha2 + ha3 + ha4);
                hb = 0.25 * (hb1 + hb2 + hb3 + hb4);
                hc = 0.25 * (hc1 + hc2 + hc3 + hc4);

                *featPtr = ha;
                *(featPtr + histStride * self->nOrientation) = hb;
                *(featPtr + 2 * histStride * self->nOrientation) = hc;
                featPtr += histStride;
            } //next k: 下一个方向
            /* 4层纹理特征 */
            featPtr = features + x + self->histWidth * y + 3 * self->nOrientation * histStride;
            *featPtr = divSqrt18 * t1;
            featPtr += histStride;
            *featPtr = divSqrt18 * t2;
            featPtr += histStride;
            *featPtr = divSqrt18 * t3;
            featPtr += histStride;
            *featPtr = divSqrt18 * t4;
            histPtr++; //指向下一个cell
        } //next x
    } //next y
#undef at
#undef atNorm
    return;
}

/**
 * @brief  快速计算反正切值, 根据输入x和y的符号自动判断象限, 输出值取值范围[0, 2pi]
 * @Param  float y:
 * @Param  float x:
 * @return float: atan(y/x)
 */
static float fastAtan2(float y, float x)
{
    float a;
    float x2 = x * x;
    float y2 = y * y;
    if (y2 <= x2)
        a = x * y / (x2 + 0.28f * y2 + (float)HOG_EPSILON) +
            (float)(x < 0 ? HOG_PI : y >= 0 ? 0 : HOG_PI * 2);
    else
        a = (float)(y >= 0 ? HOG_PI * 0.5 : HOG_PI * 1.5) -
            x * y / (y2 + 0.28f * x2 + (float)HOG_EPSILON);
    return a;
}

/**
 * @brief 获取HOG特征的列数(第一维)
 * @param self HOG描述子
 * @param width 输入图像的宽度
 * @return
 */
int getHogFeatureCols(const FHOG *self, int width)
{
    return ((width + self->cellSize / 2) / self->cellSize);
}

/**
 * @brief 获取HOG特征的行数(第二维)
 * @param self HOG描述子
 * @param height 输入图像的高度
 * @return
 */
int getHogFeatureRows(const FHOG *self, int height)
{
    return ((height + self->cellSize / 2) / self->cellSize);
}

/**
 * @brief 获取HOG特征的通道数(第3维)
 * @param self HOG描述子
 * @return
 */
int getHogFeatureChannels(const FHOG *self)
{
    return self->dimension;
}

/**
 * @brief 获取HOG特征可视化图像的宽度
 * @param self HOG描述子
 * @param width 输入图像宽度
 * @return
 */
int getHogRenderWidth(const FHOG *self, int width)
{
    return ((width + self->cellSize / 2) / self->cellSize) * self->glyphSize;
}

/**
 * @brief 获取HOG特征可视化图像的高度
 * @param self HOG描述子
 * @param height 输入图像的高度
 * @return
 */
int getHogRenderHeight(const FHOG *self, int height)
{
    return ((height + self->cellSize / 2) / self->cellSize) * self->glyphSize;
}
