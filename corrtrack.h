#ifndef CORRTRACK_H
#define CORRTRACK_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "hog.h"
#include "lpt.h"

#ifdef MAX_VAL
#undef MAX_VAL
#endif
#define MAX_VAL(x, y) ((x) > (y) ? (x) : (y))

#define FROM_CAMERA         0
#define FROM_VIDEO          1
#define FROM_IMAGESEQUENCE  2

typedef struct cRect
{
    int x;      //Left-Top x
    int y;      //Left-Top y
    int width;
    int height;
} cRect;

typedef struct cRectc
{
    int x;      //Center x
    int y;      //Center y
    int width;
    int height;
} cRectc;

typedef struct cRectp
{
    int ltx;    //Left-Top x
    int lty;    //Left-Top y
    int rbx;    //Right-Bottom x
    int rby;    //Right-Bottom y
} cRectp;

typedef struct TrackParam
{
    double transPad;
    int transPattSz;
    int transCellSz;
    int transGaussSigmaRate;
    double transLearnRate;
    bool useScale;
    double scaleMinRhoCoef;
    int scalePattSz;
    int scaleCellSz;
    int scaleGaussSigmaRate;
    double scaleLearnRate;
} TrackParam;

class CorrTrack
{
public:
    bool useScale;    
    int sourceType;
    int frameNum;
    cv::Mat frameBuf;
    cv::Mat globalApp;
    cv::Mat currentApp;
private:
    float lambda;
    float gaussCorrSigma;

    float padding;
    float transLearnRate;
    int transCellSz;
    int transPattSz;
    int transPatchNormSz;
    float transSigmaCoef;

    float scaleLearnRate;
    int scaleCellSz;
    int scalePattSz;
    int scalePatchNormSz;
    float scaleSigmaCoef;
    float rhoMinRate;
    float rhoMax;
    float rhoMin;

    cv::Rect tgtRect;
    cRectc tgtBox;
    cRectc winBox;
    float xZoom;
    float yZoom;
    int startN;

    cv::Mat tgtPatch;
    cv::Mat winPatch;
    cv::Mat lptPatch;
    cv::Mat transPatch;
    cv::Mat scalePatch;

    FHOG *transHog;
    FHOG *scaleHog;
    LPT_Grid *scaleLpt;

    cv::Mat transGaussLabelF;
    cv::Mat transHannWin;
    cv::Mat transFeat;
    cv::Mat transModelF;
    cv::Mat transAlphaF;
    cv::Mat scaleGaussLabelF;
    cv::Mat scaleHannWin;
    cv::Mat scaleFeat;
    cv::Mat scaleModelF;
    cv::Mat scaleAlphaF;

    std::vector<std::string> picSeq;
    std::vector<cv::Rect> groundTruth;
    cv::VideoCapture video;

    std::string windowName;


public:
    CorrTrack();

    CorrTrack(TrackParam *param);

    virtual ~CorrTrack();
    virtual void initParam();
    virtual void trackFromCamera(int deviceId = 0);
    virtual void trackFromVideo(const std::string videoName);
    virtual void trackFromSequence(const std::string datasetPath);

    virtual void initParam(TrackParam *param);
    virtual void initTarget(cv::Mat &frameBuf, cv::Rect &tgtRect);
    virtual void trackEachFrame(cv::Mat &frameBuf, cv::Rect &outRect);
private:
    virtual void listPicFiles(const std::string picSeqPath);
    virtual void readGroundTruth(const std::string datasetPath);
    virtual void initCamera(int deviceId = 0);
    virtual void initVideo(const std::string videoName);
    virtual void fillFrameBuf();
    virtual void initFristFrame();
    virtual void tracking();
private:
    virtual void mouseSelect(const char *window, cv::Mat &src, cv::Rect &roi);
    virtual void getGaussLabelF(cv::Mat &gaussLabelF, cv::Size &patternSz, float sigma);
    virtual void getHannWindow(cv::Mat &hannWindow, cv::Size &patternSz);
    virtual void logPolarTransform(cv::Mat &src, cv::Mat &dst, LPT_Grid *lpt);
    virtual void getPatch(cv::Mat &inImg, cv::Mat &outPatch, cRectc *rc);
    virtual void getFeatures(cv::Mat &img, cv::Mat &feat, cv::Mat &hannWin);
    virtual void getFeatures(cv::Mat &img, cv::Mat &feat, FHOG *hog);
    virtual void getFeatures(cv::Mat &img, cv::Mat &feat, cv::Mat &hannWin, FHOG *hog);
    virtual bool renderHOGFeatures(cv::Mat &feat, cv::Mat &renderImg, FHOG *hog);
    virtual void fft2(cv::Mat &feat, cv::Mat &featSpectrum);
    virtual void ifft2(cv::Mat &spectrum, cv::Mat &response);
    virtual void gaussCorrelationKernel(cv::Mat &xF, cv::Mat &yF, cv::Mat &kernelF, float sigma, bool isTrain);
    virtual double getCplxNorm(cv::Mat &src);
    virtual void getSubPixelPeak(cv::Point &maxLoc, cv::Mat &response, cv::Point2f &subPixLoc);
    virtual void train(cv::Mat &featSpectrum, cv::Mat &gaussLabelF, cv::Mat &alphaF, float sigma, float lambda);
    virtual void detect(cv::Mat &featSpectrum, cv::Mat &featModel, cv::Mat &alphaF, cv::Mat &response, cv::Point2f &pos, float sigma);
};

inline int isEven(int x)
{
    return (!(x & 0x1));
}

inline int isOdd(int x)
{
    return (x & 0x1);
}

inline bool isPower2(unsigned int n)
{
    return ((n & (n -1)) == 0);
}

inline void Rect2P(cRect *r, cRectp *rp)
{
    rp->ltx = r->x;
    rp->lty = r->y;
    rp->rbx = r->x + r->width - 1;
    rp->rby = r->y + r->height - 1;
    return;
}

inline void RectC2P(cRectc *rc, cRectp *rp)
{
    if(isOdd(rc->width))
        rp->ltx = rc->x - (rc->width >> 1);
    else
        rp->ltx = rc->x - (rc->width >> 1) + 1;
    if(isOdd(rc->height))
        rp->lty = rc->y - (rc->height >> 1);
    else
        rp->lty = rc->y - (rc->height >> 1) + 1;
    rp->rbx = rc->x + (rc->width >> 1);
    rp->rby = rc->y + (rc->height >> 1);
    return;
}

inline void Rect2C(cv::Rect &r, cRectc *rc)
{
    if(isOdd(r.width))
        rc->x = r.x + (r.width >> 1);
    else
        rc->x = r.x + (r.width >> 1) - 1;
    if(isOdd(r.height))
        rc->y = r.y + (r.height >> 1);
    else
        rc->y = r.y + (r.height >> 1) - 1;
    rc->width = r.width;
    rc->height = r.height;
}


#endif // CORRTRACK_H
