#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "lpt.h"


static __inline unsigned char trunc2byte(int value);
static __inline unsigned char bilinear(const unsigned char *pixel, unsigned int u_8, unsigned int v_8);
static __inline unsigned char bilinear_rightBorder(const unsigned char *pixel, unsigned int u_8, unsigned int v_8);
static __inline unsigned char bilinear_bottomBorder(const unsigned char *pixel, unsigned int u_8, unsigned int v_8);


/**
 * @brief 新建LPT_Grid结构
 * @param imgWidth 源图像宽度
 * @param imgHeight 源图像高度
 * @param gridWidth 目标图像的宽度
 * @param gridHeight 目标图像的高度
 * @param rhoMinRate 最小极径系数
 * @return LPT_Grid结构
 * 当需要进行一系列相同尺寸图像的LPT变换操作时, 仅仅新建一个LPT_Grid结构即可.
 * 该结构的作用即是提前计算变换前后的两幅图像之间的像素映射关系, 具有加速的效果.
 */
LPT_Grid *newLptGrid(int imgWidth, int imgHeight, int gridWidth, int gridHeight, float rhoMinRate)
{
    float centerX = imgWidth * 0.5;
    float centerY = imgHeight * 0.5;
    float rhoMax = sqrt(imgWidth * imgWidth + imgHeight * imgHeight) * 0.5;
    float rhoMin = rhoMax * rhoMinRate;
    float logRhoMax = (float)log(rhoMax + LPT_EPSILON);
    float logRhoMin = (float)log(rhoMin + LPT_EPSILON);
    float inc;
    float *rho, *theta, *cosT, *sinT;
    LPT_Grid *lpt;
    int i, j;
    assert(rhoMinRate >= 0 && rhoMinRate < 1);
    rho = (float*)malloc(sizeof(float) * gridWidth);
    theta = (float*)malloc(sizeof(float) * gridHeight);
    cosT = (float*)malloc(sizeof(float) * gridHeight);
    sinT = (float*)malloc(sizeof(float) * gridHeight);
    lpt = (LPT_Grid*)malloc(sizeof(LPT_Grid));
    assert(lpt != NULL && rho != NULL && theta != 0);
    inc = (logRhoMax - logRhoMin) / (gridWidth - 1);
    rho[0] = logRhoMin;
    for(i = 1; i < gridWidth; i++)
        rho[i] = rho[i-1] + inc;
    for(i = 0; i < gridWidth; i++)
        rho[i] = (float)exp(rho[i]);
    inc = (LPT_PI * 2) / gridHeight;
    theta[0] = -LPT_PI;
    cosT[0] = (float)cos(theta[0]);
    sinT[0] = (float)sin(theta[0]);
    for(i = 1; i < gridHeight; i++)
    {
        theta[i] = theta[i-1] + inc;
        cosT[i] = (float)cos(theta[i]);
        sinT[i] = (float)sin(theta[i]);
    }
    lpt->imgWidth = imgWidth;
    lpt->imgHeight = imgHeight;
    lpt->rho = gridWidth;
    lpt->theta = gridHeight;
    lpt->rhoMinRate = rhoMinRate;
    lpt->xGrid = (int*)malloc(sizeof(int) * gridWidth * gridHeight);
    lpt->yGrid = (int*)malloc(sizeof(int) * gridWidth * gridHeight);
    assert(lpt->xGrid != NULL && lpt->yGrid != NULL);
    for(j = 0; j < gridHeight; j++)
    {
        int *px = lpt->xGrid + j * gridWidth;
        int *py = lpt->yGrid + j * gridWidth;
        for(i = 0; i < gridWidth; i++)
        {
            px[i] = (int)((cosT[j] * rho[i] + centerX) * 65536); /* 放大65536倍, 避免浮点运算 */
            py[i] = (int)((sinT[j] * rho[i] + centerY) * 65536); /* 同上 */
        }
    }
    free(rho);
    free(theta);
    free(cosT);
    free(sinT);
    return lpt;
}

/**
 * @brief 释放LPT_Grid结构
 * @param lpt
 */
void freeLptGrid(LPT_Grid *lpt)
{
    if(lpt->xGrid)
    {
        free(lpt->xGrid);
        lpt->xGrid = NULL;
    }
    if(lpt->yGrid)
    {
        free(lpt->yGrid);
        lpt->yGrid = NULL;
    }
    if(lpt)
        free(lpt);
    return;
}

/**
 * @brief 图像的对数极坐标变换
 * @param src 源图像
 * @param dst 目标图像
 * @param lpt LPT变换所需的插值网格结构
 * 此函数对于邻近像素插值的权重采用整形运算, 避免了大量浮点运算对整体计算速度的拖累.
 * 每个像素都有3个近邻像素与其共同对目标像素作贡献, 4个像素之间的空间关系如下:
 * +---+---+
 * | 0 | 2 |
 * +---+---+
 * | 1 | 3 |
 * +---+---+
 * 根据插值网格中定义的已放大65536(左移16位)的(x, y)坐标, 通过整数的右移操作产生
 * 的截断效果, 可以得到0号像素的正确坐标值, 附带而来的还有0号像素在x方向与y方向上
 * 与邻近像素的权值. 据此, 可以完成像素的双线性插值操作
 */
void logPolar(const unsigned char *src, unsigned char *dst, LPT_Grid *lpt)
{
    int j, i;
    int x, y; /* 源图像的坐标 */
    int xBorderFlag, yBorderFlag; /* 源图像坐标越界标志 */
    int nanFlag; /* 源图像坐标为负值标志(内存泄露) */
    unsigned int u_8, v_8; /* 源图像中像素线性插值的权重(放大256倍的) */
    int ws = lpt->imgWidth;
    int hs = lpt->imgHeight;
    unsigned char *pd;
    unsigned char pixel[4] = {0, 0, 0, 0};
    for(j = 0; j < lpt->theta; j++)
    {
        int *px = lpt->xGrid + j * lpt->rho;
        int *py = lpt->yGrid + j * lpt->rho;
        pd = dst + j * lpt->rho;
        for(i = 0; i < lpt->rho; i++)
        {
            x = px[i] >> 16;
            y = py[i] >> 16;
            u_8 = (px[i] & 0xFFFF) >> 8;
            v_8 = (py[i] & 0xFFFF) >> 8;
            xBorderFlag = (x == (ws - 1)) ? 1 : 0;
            yBorderFlag = (y == (hs - 1)) ? 1 : 0;
            nanFlag = (x < 0 || y < 0 || x >= ws || y >= hs) ? 1 : 0;
            if(xBorderFlag && yBorderFlag && (!nanFlag)) /* 同时越右边界和下边界 */
                pd[i] = *(src + y * ws + x);
            else if(xBorderFlag && (!yBorderFlag) && (!nanFlag)) /* 越右边界 */
            {
                pixel[0] = *(src + y * ws + x);
                pixel[1] = *(src + (y + 1) * ws + x);
                pd[i] = bilinear_rightBorder(pixel, u_8, v_8);
            }
            else if((!xBorderFlag) && yBorderFlag && (!nanFlag)) /* 越下边界 */
            {
                pixel[0] = *(src + y * ws + x);
                pixel[2] = *(src + y * ws + x + 1);
                pd[i] = bilinear_bottomBorder(pixel, u_8, v_8);
            }
            else if(!nanFlag) /* 未越界 */
            {
                pixel[0] = *(src + y * ws + x);
                pixel[1] = *(src + (y + 1) * ws + x);
                pixel[2] = *(src + y * ws + x + 1);
                pixel[3] = *(src + (y + 1) * ws + x + 1);
                pd[i] = bilinear(pixel, u_8, v_8);
            }
            else
                pd[i] = 0;
        }
    }
    return;
}

/**
 * @brief 把整形数据截断在[0, 255]之间成为unsigned char型
 * @param Value: 输入int型数据
 * @return
 */
static __inline unsigned char trunc2byte(int value)
{
    return ((value | ((int)(255 - value) >> 31) ) & ~((int)value >> 31));
}

/**
 * @brief 图像快速双线性插值(不越界情况)
 * @param pixel: 相邻的4个像素值
 * @param u_8: 距左上角像素的水平距离(已放大256倍)
 * @param v_8: 距左上角像素的垂直距离(已放大256倍)
 * @return 内插的像素值
 */
static __inline unsigned char bilinear(const unsigned char *pixel, unsigned int u_8, unsigned int v_8)
{
    unsigned int s3_16 = (u_8 * v_8);
    unsigned int s2_16 = (u_8 << 8) - s3_16;
    unsigned int s1_16 = (v_8 << 8) - s3_16;
    unsigned int s0_16 = (1 << 16) - s1_16 - s2_16 - s3_16;
    unsigned int p = pixel[0] * s0_16 + pixel[1] * s1_16 + pixel[2] * s2_16 + pixel[3] * s3_16;
    return trunc2byte((int)(p >> 16));
}

/**
 * @brief 图像快速双线性插值(越右边界一个像素)
 * @param pixel: 相邻的4个像素值(仅用到其中2个)
 * @param u_8: 距左上角像素的水平距离(已放大256倍)
 * @param v_8: 距左上角像素的垂直距离(已放大256倍)
 * @return 内插的像素值
 */
static __inline unsigned char bilinear_rightBorder(const unsigned char *pixel, unsigned int u_8, unsigned int v_8)
{
    unsigned int s1_16 = (256 - u_8) * v_8;
    unsigned int s0_16 = (256 - u_8) * (256 - v_8);
    unsigned int p = pixel[0] * s0_16 + pixel[1] * s1_16;
    return trunc2byte((int)(p >> 16));
}

/**
 * @brief 图像快速双线性插值(越下边界一个像素)
 * @param pixel: 相邻的4个像素值(仅用到其中2个)
 * @param u_8: 距左上角像素的水平距离(已放大256倍)
 * @param v_8: 距左上角像素的垂直距离(已放大256倍)
 * @return 内插的像素值
 */
static __inline unsigned char bilinear_bottomBorder(const unsigned char *pixel, unsigned int u_8, unsigned int v_8)
{
    unsigned int s0_16 = (256 - u_8) * (256 - v_8);
    unsigned int s2_16 = (256 - v_8) * u_8;
    unsigned int p = pixel[0] * s0_16 + pixel[2] * s2_16;
    return trunc2byte((int)(p >> 16));
}
