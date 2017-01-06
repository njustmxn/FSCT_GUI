#include <iostream>
#include <fstream>
#include <istream>
#include <string>
#include <vector>
#include <string>
#include <direct.h>
#include <io.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "corrtrack.h"

using namespace std;
using namespace cv;

static const float DIV255 = 0.0039215f;
static cv::Point _LEFTUP;       //左上角点
static cv::Point _RIGHTDOWN;    //右下角点
static bool isDrawing;          //正在画框的标志量
static void onMouse(int mouseEvent, int x, int y, int, void*)
{
    switch (mouseEvent)
    {
    case CV_EVENT_LBUTTONDOWN:
        _LEFTUP = Point(x, y);
        _RIGHTDOWN = Point(x, y);
        isDrawing = true;
        break;
    case CV_EVENT_MOUSEMOVE:
        if (isDrawing)
            _RIGHTDOWN = Point(x, y);
        break;
    case CV_EVENT_LBUTTONUP:
        _RIGHTDOWN = Point(x, y);
        isDrawing = false;
        break;
    }
    return;
}

CorrTrack::CorrTrack()
{
    initParam();
}

CorrTrack::CorrTrack(TrackParam *param)
{
    initParam(param);
}

CorrTrack::~CorrTrack()
{
    freeHogDescriptor(transHog);
    if(useScale)
    {
        freeHogDescriptor(scaleHog);
        freeLptGrid(scaleLpt);
    }
}

void CorrTrack::initParam()
{
    windowName = "Fast Scale-Adaptive Correlation Tracking";
    frameNum = 0;
    lambda = 0.0001;
    gaussCorrSigma = 0.5;
    padding = 1.5;
    transCellSz = 4;
    transPattSz = 32;    
    transLearnRate = 0.02;
    transSigmaCoef = 1.0/26;
    if(useScale)
    {
        scaleCellSz = 4;
        scalePattSz = 32;
        scaleLearnRate = 0.02;
        scaleSigmaCoef = 1.0/24;
        rhoMinRate = 0.2;
    }
}

void CorrTrack::initParam(TrackParam *param)
{
    windowName = "Fast Scale-Adaptive Correlation Tracking";
    frameNum = 0;
    lambda = 0.0001;
    gaussCorrSigma = 0.5;
    padding = param->transPad;
    transCellSz = param->transCellSz;
    transPattSz = param->transPattSz;
    transLearnRate = param->transLearnRate;
    transSigmaCoef = 1.0 / param->transGaussSigmaRate;
    if(useScale = param->useScale)
    {
        scaleCellSz = param->scaleCellSz;
        scalePattSz = param->scalePattSz;
        scaleLearnRate = param->scaleLearnRate;
        scaleSigmaCoef = 1.0 / param->scaleGaussSigmaRate;
        rhoMinRate = param->scaleMinRhoCoef;
    }
}

void CorrTrack::initCamera(int deviceId)
{
    video.open(deviceId);
    if(!video.isOpened())
        perror("无法初始化摄像头, 请确认该设备已正确安装!\n");
    return;
}

void CorrTrack::initVideo(const string videoName)
{
    video.open(videoName);
    if(!video.isOpened())
        perror("无法播放该视频, 请确认解码库已正确安装!\n");
    return;
}

void CorrTrack::trackFromCamera(int deviceId)
{
    initCamera(deviceId);
    initParam();
    sourceType = FROM_CAMERA;
    initFristFrame();
    tracking();
    return;
}

void CorrTrack::trackFromVideo(const string videoName)
{
    initVideo(videoName);
    initParam();
    sourceType = FROM_VIDEO;
    initFristFrame();
    tracking();
    return;
}

void CorrTrack::trackFromSequence(const string datasetPath)
{
    readGroundTruth(datasetPath);
    string picSeqPath = datasetPath;
    if(picSeqPath[picSeqPath.length()-1] != '\\')
        picSeqPath += '\\';
    picSeqPath += "img\\";
    listPicFiles(picSeqPath);
    initParam();
    sourceType = FROM_IMAGESEQUENCE;
    initFristFrame();
    tracking();
    return;
}

void CorrTrack::fillFrameBuf()
{
    if (sourceType == FROM_CAMERA || sourceType == FROM_VIDEO)
    {
        video >> frameBuf;
        frameNum++;
    }
    else
    {
        if(frameNum >= picSeq.size()-1 || frameNum >= groundTruth.size()-1)
            frameBuf.deallocate();
        else
            frameBuf = imread(picSeq[frameNum++]);
    }
}

void CorrTrack::initFristFrame()
{
    Mat I;
    namedWindow(windowName);
    if(sourceType == FROM_CAMERA || sourceType == FROM_VIDEO)
    {
        while(1)
        {
            video >> frameBuf;
            string text("Press Enter to stop, draw rectangle on target, then press Enter...");
            putText(frameBuf, text, Point(5, 40), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 1);
            imshow(windowName, frameBuf);
            char key = waitKey(33);
            if(key == 13)
                break;
        }
    }
    else
    {
        frameBuf = imread(picSeq[frameNum++]);
        imshow(windowName, frameBuf);
    }
    if(frameBuf.channels() == 3)
        cvtColor(frameBuf, I, CV_BGR2GRAY);
    else
        frameBuf.copyTo(I);
    if(sourceType == FROM_CAMERA || sourceType == FROM_VIDEO)
        mouseSelect(windowName.c_str(), I, tgtRect);
    else
        tgtRect = groundTruth[frameNum];
    Rect2C(tgtRect, &tgtBox);
    winBox.x = tgtBox.x;
    winBox.y = tgtBox.y;
    winBox.width = floor(1.0 * tgtBox.width * (padding + 1));
    winBox.height = floor(1.0 * tgtBox.height * (padding + 1));
    xZoom = 1.0 * (winBox.width - 1) / (transPattSz - 1);
    yZoom = 1.0 * (winBox.height - 1) / (transPattSz - 1);
    transPatchNormSz = transPattSz * transCellSz;
    transHog = newHogDescriptor(transCellSz, 9, 0, 0);
    Size patSz(transPattSz, transPattSz);
    getHannWindow(transHannWin, patSz);
    getGaussLabelF(transGaussLabelF, patSz, transSigmaCoef * transPattSz);
    getPatch(I, winPatch, &winBox);
    resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
    getFeatures(transPatch, transFeat, transHannWin, transHog);
    fft2(transFeat, transModelF);
    train(transModelF, transGaussLabelF, transAlphaF, gaussCorrSigma, lambda);

    if(useScale)
    {
        scalePatchNormSz = scaleCellSz * scalePattSz;
        rhoMax = log(std::sqrt(2.0) * 0.5 * scalePattSz);
        rhoMin = log(0.5 * scalePattSz * rhoMinRate);
        scaleHog = newHogDescriptor(scaleCellSz, 9, 0, 0);
        scaleLpt = newLptGrid(scalePatchNormSz, scalePatchNormSz,
                              scalePatchNormSz, scalePatchNormSz, rhoMinRate);
        lptPatch.create(scalePatchNormSz, scalePatchNormSz, CV_8U);
        Size patSz(scalePattSz, scalePattSz);
        getHannWindow(scaleHannWin, patSz);
        getGaussLabelF(scaleGaussLabelF, patSz, scalePattSz * scaleSigmaCoef);
        getPatch(I, tgtPatch, &tgtBox);
        resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
        logPolarTransform(lptPatch, scalePatch, scaleLpt);
        getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
        fft2(scaleFeat, scaleModelF);
        train(scaleModelF, scaleGaussLabelF, scaleAlphaF, gaussCorrSigma, lambda);
    }
    rectangle(frameBuf, tgtRect, Scalar(0,0,255), 2);
    imshow(windowName, frameBuf);
    waitKey(5);
    return;
}

void CorrTrack::initTarget(Mat &frameBuf, Rect &tgtRect)
{
    Mat I;
    if(frameBuf.channels() == 3)
        cvtColor(frameBuf, I, CV_BGR2GRAY);
    else
        frameBuf.copyTo(I);
    Rect2C(tgtRect, &tgtBox);
    winBox.x = tgtBox.x;
    winBox.y = tgtBox.y;
    winBox.width = floor(1.0 * tgtBox.width * (padding + 1));
    winBox.height = floor(1.0 * tgtBox.height * (padding + 1));
    xZoom = 1.0 * (winBox.width - 1) / (transPattSz - 1);
    yZoom = 1.0 * (winBox.height - 1) / (transPattSz - 1);
    transPatchNormSz = transPattSz * transCellSz;
    transHog = newHogDescriptor(transCellSz, 9, 0, 0);
    Size patSz(transPattSz, transPattSz);
    getHannWindow(transHannWin, patSz);
    getGaussLabelF(transGaussLabelF, patSz, transSigmaCoef * transPattSz);
    getPatch(I, winPatch, &winBox);
    resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
    getFeatures(transPatch, transFeat, transHannWin, transHog);
    fft2(transFeat, transModelF);
    train(transModelF, transGaussLabelF, transAlphaF, gaussCorrSigma, lambda);
    if(useScale)
    {
        scalePatchNormSz = scaleCellSz * scalePattSz;
        rhoMax = log(std::sqrt(2.0) * 0.5 * scalePattSz);
        rhoMin = log(0.5 * scalePattSz * rhoMinRate);
        scaleHog = newHogDescriptor(scaleCellSz, 9, 0, 0);
        scaleLpt = newLptGrid(scalePatchNormSz, scalePatchNormSz,
                              scalePatchNormSz, scalePatchNormSz, rhoMinRate);
        lptPatch.create(scalePatchNormSz, scalePatchNormSz, CV_8U);
        Size patSz(scalePattSz, scalePattSz);
        getHannWindow(scaleHannWin, patSz);
        getGaussLabelF(scaleGaussLabelF, patSz, scalePattSz * scaleSigmaCoef);
        getPatch(I, tgtPatch, &tgtBox);
        resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
        logPolarTransform(lptPatch, scalePatch, scaleLpt);
        getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
        fft2(scaleFeat, scaleModelF);
        train(scaleModelF, scaleGaussLabelF, scaleAlphaF, gaussCorrSigma, lambda);
    }
    transPatch.copyTo(globalApp);
}

void CorrTrack::trackEachFrame(Mat &frameBuf, Rect &outRect)
{
    Mat I;
    cRectp rp = {0, 0, 0, 0};
    if(frameBuf.empty())
        return;
    if(frameBuf.channels() == 3)
        cvtColor(frameBuf, I, CV_BGR2GRAY);
    else
        frameBuf.copyTo(I);
    getPatch(I, winPatch, &winBox);
    resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
    getFeatures(transPatch, transFeat, transHannWin, transHog);
    Mat transXF, transResponse;
    Point2f resPos;
    fft2(transFeat, transXF);
    detect(transXF, transModelF, transAlphaF, transResponse, resPos, gaussCorrSigma);

    RectC2P(&winBox, &rp);
    winBox.x = floor((resPos.x) * xZoom + rp.ltx);
    winBox.y = floor((resPos.y) * yZoom + rp.lty);
    tgtBox.x = winBox.x;
    tgtBox.y = winBox.y;

    if(useScale)
    {
        getPatch(I, tgtPatch, &tgtBox);
        resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
        logPolarTransform(lptPatch, scalePatch, scaleLpt);
        getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
        Mat scaleXF, scaleResponse;
        float scale;
        fft2(scaleFeat, scaleXF);
        detect(scaleXF, scaleModelF, scaleAlphaF, scaleResponse, resPos, gaussCorrSigma);
        scale = exp(-log(rhoMinRate) * (resPos.x - (scalePattSz - 1) * 0.5) / scalePattSz);
        tgtBox.width = cvRound(1.0 * tgtBox.width * scale);
        tgtBox.height = cvRound(1.0 * tgtBox.height * scale);
        winBox.width = cvRound(1.0 * tgtBox.width * (padding + 1));
        winBox.height = cvRound(1.0 * tgtBox.height * (padding + 1));
        xZoom = 1.0 * (winBox.width - 1) / (transPattSz - 1);
        yZoom = 1.0 * (winBox.height - 1) / (transPattSz - 1);
    }

    getPatch(I, winPatch, &winBox);
    resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
    getFeatures(transPatch, transFeat, transHannWin, transHog);
    Mat transModelF_new, transAlphaF_new;
    fft2(transFeat, transModelF_new);
    train(transModelF_new, transGaussLabelF, transAlphaF_new, gaussCorrSigma, lambda);
    accumulateWeighted(transModelF_new, transModelF, transLearnRate);
    accumulateWeighted(transAlphaF_new, transAlphaF, transLearnRate);
    Mat tmp;
    globalApp.convertTo(tmp, CV_32F);
    accumulateWeighted(transPatch, tmp, transLearnRate);
    tmp.convertTo(globalApp, CV_8U);
    transPatch.copyTo(currentApp);

    if(useScale)
    {
        getPatch(I, tgtPatch, &tgtBox);
        resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
        logPolarTransform(lptPatch, scalePatch, scaleLpt);
        getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
        Mat scaleModelF_new, scaleAlphaF_new;
        fft2(scaleFeat, scaleModelF_new);
        train(scaleModelF_new, scaleGaussLabelF, scaleAlphaF_new, gaussCorrSigma, lambda);
        accumulateWeighted(scaleModelF_new, scaleModelF, scaleLearnRate);
        accumulateWeighted(scaleAlphaF_new, scaleAlphaF, scaleLearnRate);
    }
    tgtRect.x = cvRound(tgtBox.x - tgtBox.width * 0.5);
    tgtRect.y = cvRound(tgtBox.y - tgtBox.height * 0.5);
    tgtRect.width = tgtBox.width;
    tgtRect.height = tgtBox.height;
    outRect = tgtRect;

}

void CorrTrack::listPicFiles(const string picSeqPath)
{
    string path = picSeqPath;
    if(path[path.length()-1] != '\\')
        path += '\\';
    _chdir(path.c_str());
    picSeq.clear();
    char *filespec = "*.jpg";
    //首先查找dir中符合要求的文件
    long hFile;
    _finddata_t fileinfo;
    if ((hFile=_findfirst(filespec, &fileinfo)) != -1)
    {
        do
        {
            //检查是不是目录
            //如果不是,则进行处理
            if (!(fileinfo.attrib & _A_SUBDIR))
            {
                char filename[_MAX_PATH];
                strcpy(filename, path.c_str());
                strcat(filename, fileinfo.name);
                picSeq.push_back(filename);
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
    return;
}

void CorrTrack::readGroundTruth(const string datasetPath)
{
    string path = datasetPath;
    if(path[path.length()-1] != '\\')
        path += '\\';
    path += "groundtruth_rect.txt";
    char gt[_MAX_PATH];
    strcpy(gt, path.c_str());
    FILE *fpgt;
    Rect tmp;
    if((fpgt = fopen(gt, "r")) == NULL)
        perror("无法读取GroundTruth文件, 请确认是否存在!\n");
    bool commaGt = false;
    while(!feof(fpgt))
    {
        if(fgetc(fpgt) == ',')
        {
            commaGt = true;
            break;
        }
    }
    fseek(fpgt, 0L, SEEK_SET);
    if(commaGt)
    {
        while(4 == fscanf(fpgt, "%d,%d,%d,%d", &tmp.x, &tmp.y, &tmp.width, &tmp.height))
            groundTruth.push_back(tmp);
    }
    else
    {
        while(4 == fscanf(fpgt, "%d%d%d%d", &tmp.x, &tmp.y, &tmp.width, &tmp.height))
            groundTruth.push_back(tmp);
    }
    fclose(fpgt);
    return;
}

void CorrTrack::tracking()
{
    Mat I;
    double t, time = 0;
    double fps = 0;    
    while(1)
    {
        cRectp rp = {0, 0, 0, 0};
        fillFrameBuf();
        if(frameBuf.empty())
            break;
        if(sourceType == FROM_IMAGESEQUENCE && (frameNum >= groundTruth.size()-1 || frameNum >= picSeq.size()-1))
            break;
        if(frameBuf.channels() == 3)
            cvtColor(frameBuf, I, CV_BGR2GRAY);
        else
            frameBuf.copyTo(I);
        t = (double)getTickCount();
        getPatch(I, winPatch, &winBox);
        resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
        getFeatures(transPatch, transFeat, transHannWin, transHog);
        Mat transXF, transResponse;
        Point2f resPos;
        fft2(transFeat, transXF);
        detect(transXF, transModelF, transAlphaF, transResponse, resPos, gaussCorrSigma);

        RectC2P(&winBox, &rp);
        winBox.x = floor((resPos.x) * xZoom + rp.ltx);
        winBox.y = floor((resPos.y) * yZoom + rp.lty);
        tgtBox.x = winBox.x;
        tgtBox.y = winBox.y;

        if(useScale)
        {
            getPatch(I, tgtPatch, &tgtBox);
            resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
            logPolarTransform(lptPatch, scalePatch, scaleLpt);
            getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
            Mat scaleXF, scaleResponse;
            float scale;
            fft2(scaleFeat, scaleXF);
            detect(scaleXF, scaleModelF, scaleAlphaF, scaleResponse, resPos, gaussCorrSigma);
            scale = exp(-log(rhoMinRate) * (resPos.x - (scalePattSz - 1) * 0.5) / scalePattSz);
            tgtBox.width = cvRound(1.0 * tgtBox.width * scale);
            tgtBox.height = cvRound(1.0 * tgtBox.height * scale);
            winBox.width = cvRound(1.0 * tgtBox.width * (padding + 1));
            winBox.height = cvRound(1.0 * tgtBox.height * (padding + 1));
            xZoom = 1.0 * (winBox.width - 1) / (transPattSz - 1);
            yZoom = 1.0 * (winBox.height - 1) / (transPattSz - 1);
        }

        getPatch(I, winPatch, &winBox);
        resize(winPatch, transPatch, Size(transPatchNormSz, transPatchNormSz));
        getFeatures(transPatch, transFeat, transHannWin, transHog);
        Mat transModelF_new, transAlphaF_new;
        fft2(transFeat, transModelF_new);
        train(transModelF_new, transGaussLabelF, transAlphaF_new, gaussCorrSigma, lambda);
        accumulateWeighted(transModelF_new, transModelF, transLearnRate);
        accumulateWeighted(transAlphaF_new, transAlphaF, transLearnRate);

        if(useScale)
        {
            getPatch(I, tgtPatch, &tgtBox);
            resize(tgtPatch, lptPatch, Size(scalePatchNormSz, scalePatchNormSz));
            logPolarTransform(lptPatch, scalePatch, scaleLpt);
            getFeatures(scalePatch, scaleFeat, scaleHannWin, scaleHog);
            Mat scaleModelF_new, scaleAlphaF_new;
            fft2(scaleFeat, scaleModelF_new);
            train(scaleModelF_new, scaleGaussLabelF, scaleAlphaF_new, gaussCorrSigma, lambda);
            accumulateWeighted(scaleModelF_new, scaleModelF, scaleLearnRate);
            accumulateWeighted(scaleAlphaF_new, scaleAlphaF, scaleLearnRate);
        }

        tgtRect.x = cvRound(tgtBox.x - tgtBox.width * 0.5);
        tgtRect.y = cvRound(tgtBox.y - tgtBox.height * 0.5);
        tgtRect.width = tgtBox.width;
        tgtRect.height = tgtBox.height;

        t = (double)getTickCount() - t;
        time += t;
        fps = frameNum / (time / getTickFrequency());

        if(sourceType == FROM_IMAGESEQUENCE)
            rectangle(frameBuf, groundTruth[frameNum], Scalar(0,255,0), 2);
        rectangle(frameBuf, tgtRect, Scalar(0,0,255), 2);

        char num[30];
        sprintf(num, "#%d", frameNum);
        string text(num);        
        putText(frameBuf, text, Point(5, 20), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 1);
        memset(num, 0, 20);
        sprintf(num, "FPS: %03.1f", fps);
        string text1(num);
        putText(frameBuf, text1, Point(5, 40), FONT_HERSHEY_PLAIN, 1, Scalar(0,0,255), 1);

        imshow(windowName, frameBuf);
        char key = waitKey(5);
        if(key == 13)
            break;        
    }
    destroyAllWindows();
    return;
}

void CorrTrack::mouseSelect(const char *window, Mat &src, Rect &roi)
{
    _LEFTUP = Point(0,0);
    _RIGHTDOWN = Point(0,0);
    isDrawing = false;
    Mat srcCpy;
    printf("\n********Control Information********\n");
    printf("Drag mouse to draw rectangle, and then:\n");
    printf("1. move up:    W\n");
    printf("2. move down:  S\n");
    printf("3. move left:  A\n");
    printf("4. move right: D\n");
    printf("5. zoom in:    1\n");
    printf("6. zoom out:   2\n");
    printf("7. confirm:    Enter\n");
    printf("***********************************\n\n");
    while (true)
    {
        setMouseCallback(window, onMouse);
        src.copyTo(srcCpy);
        if( src.channels() == 3)
            rectangle(srcCpy, _LEFTUP, _RIGHTDOWN, Scalar(0, 0, 255));
        else
            rectangle(srcCpy, _LEFTUP, _RIGHTDOWN, Scalar(0));
        imshow(window, srcCpy);
        printf("Point : (%d, %d) to (%d, %d) \t\tSize : (%d, %d)\r", _LEFTUP.x, _LEFTUP.y,
               _RIGHTDOWN.x, _RIGHTDOWN.y, _RIGHTDOWN.y - _LEFTUP.y, _RIGHTDOWN.x - _LEFTUP.x);
        char key = waitKey(5);
        switch (key)
        {
        //ROI平移操作,使用键盘方向键
        case 'a':
            (_LEFTUP.x > 0) ? (_LEFTUP.x--) : (_LEFTUP.x = 0);
            _RIGHTDOWN.x--;
            break;
        case 's':
            _LEFTUP.y++;
            (_RIGHTDOWN.y < src.rows -1) ? (_RIGHTDOWN.y++) : (_RIGHTDOWN.y = src.rows - 1);
            break;
        case 'd':
            _LEFTUP.x++;
            (_RIGHTDOWN.x < src.cols -1) ? (_RIGHTDOWN.x++) : (_RIGHTDOWN.x = src.cols - 1);
            break;
        case 'w':
            (_LEFTUP.y > 0) ? (_LEFTUP.y--) : (_LEFTUP.y = 0);
            _RIGHTDOWN.y--;
            break;
        //ROI放大和缩小，pageup键放大，pagedown键缩小
        case '1':
            (_LEFTUP.x > 0) ? (_LEFTUP.x--) : (_LEFTUP.x = 0);
            (_LEFTUP.y > 0) ? (_LEFTUP.y--) : (_LEFTUP.y = 0);
            (_RIGHTDOWN.x < src.cols - 1) ? (_RIGHTDOWN.x++) : (_RIGHTDOWN.x = src.cols - 1);
            (_RIGHTDOWN.y < src.rows - 1) ? (_RIGHTDOWN.y++) : (_RIGHTDOWN.y = src.rows - 1);
            break;
        case '2':
            _LEFTUP.x++;
            _LEFTUP.y++;
            _RIGHTDOWN.x--;
            _RIGHTDOWN.y--;
            break;
        //回车确定最终ROI区域的截取，并将其保存下来
        case 13:
            int w = std::abs(_RIGHTDOWN.x - _LEFTUP.x);
            if(w % 4) w += (4 - w % 4); //保证所选区域宽度为4的倍数
            int h = std::abs(_RIGHTDOWN.y - _LEFTUP.y);
            if(h % 4) h += (4 - h % 4); //保证所选区域高度为4的倍数
            roi = Rect(_LEFTUP.x, _LEFTUP.y, w, h);
            roi &= Rect(0, 0, src.cols, src.rows);
            printf("Selected target size (width, height) : (%d, %d)\n\n", w, h);
            break;
        }
        if (key == 13)
            break;
    }
    return;
}

void CorrTrack::getGaussLabelF(Mat &gaussLabelF, Size &patternSz, float sigma)
{
    int i, j;
    float xhalf = patternSz.width * 0.5;
    float yhalf = patternSz.height * 0.5;
    float scale = -0.5 / (sigma * sigma);
    Mat gaussLabel(patternSz, CV_32F);
    if(gaussLabelF.data == NULL)
        gaussLabelF.create(patternSz, CV_32FC2);
    for(i = 0; i < patternSz.height; i++)
    {
        float *p = gaussLabel.ptr<float>(i, 0);
        float y = i + 0.5 - yhalf;
        for(j = 0; j < patternSz.width; j++)
        {
            float x = j + 0.5 - xhalf;
            p[j] = (float)exp(scale * (x * x + y * y));
        }
    }
    dft(gaussLabel, gaussLabelF, DFT_COMPLEX_OUTPUT);
    return;
}

void CorrTrack::getHannWindow(Mat &hannWindow, Size &patternSz)
{
    int i, j;
    if(hannWindow.data == NULL)
        hannWindow.create(patternSz, CV_32F);
    for(i = 0; i < patternSz.height; i++)
    {
        float *p = hannWindow.ptr<float>(i, 0);
        float y = 1 - cos(2.0 * CV_PI * (i + 0.5) / patternSz.height);
        for(j = 0; j < patternSz.width; j++)
        {
            float x = 1 - cos(2.0 * CV_PI * (j + 0.5) / patternSz.width);
            p[j] = 0.25 * x * y;
        }
    }
    return;
}

void CorrTrack::logPolarTransform(Mat &src, Mat &dst, LPT_Grid *lpt)
{
    if(dst.data == NULL)
        dst.create(lpt->theta, lpt->rho, CV_8U);
    assert(lpt != NULL && dst.rows == lpt->theta && dst.cols == lpt->rho);
    logPolar(src.data, dst.data, lpt);
    return;
}

void CorrTrack::getPatch(Mat &inImg, Mat &outPatch, cRectc *rc)
{
    assert(rc->x >= 0 && rc->x < inImg.cols
           && rc->y >= 0 && rc->y < inImg.rows
           && rc->width >= 1 && rc->height >= 1);
    cRectp rp = {0, 0, 0, 0};
    RectC2P(rc, &rp);
    int lep = (rp.ltx > 0) ? 0 : abs(rp.ltx);
    int tep = (rp.lty > 0) ? 0 : abs(rp.lty);
    int rep = ((rp.rbx + 1 - inImg.cols) < 0) ? 0 : (rp.rbx + 1 - inImg.cols);
    int bep = ((rp.rby + 1 - inImg.rows) < 0) ? 0 : (rp.rby + 1 - inImg.rows);
    Rect roi(rp.ltx + lep, rp.lty + tep, rc->width - lep - rep, rc->height - tep - bep);
    Mat tmp(inImg, roi);
    if(outPatch.data == NULL)
        outPatch.create(rc->height, rc->width, CV_8U);
    copyMakeBorder(tmp, outPatch, tep, bep, lep, rep, BORDER_REPLICATE);
    return;
}

void CorrTrack::getFeatures(Mat &img, Mat &feat, Mat &hannWin)
{
    assert(img.rows == hannWin.rows && img.cols == hannWin.cols);
    img.convertTo(feat, CV_32F);
    for(int i = 0; i < feat.rows; i++)
    {
        float *pf = feat.ptr<float>(i, 0);
        float *ph = hannWin.ptr<float>(i, 0);
        for(int j = 0; j < feat.cols; j++)
        {
            pf[j] *= DIV255;
            pf[j] -= 0.5;
            pf[j] *= ph[j];
        }
    }
    return;
}

void CorrTrack::getFeatures(Mat &img, Mat &feat, FHOG *hog)
{
    if(feat.data == NULL)
    {
        int rows = getHogFeatureRows(hog, img.rows);
        int cols = getHogFeatureCols(hog, img.cols);
        int channels = getHogFeatureChannels(hog);
        int sz[3] = {channels, rows, cols};
        feat.create(3, sz, CV_32F);
    }
    float *featPtr = feat.ptr<float>(0, 0, 0);
    calcHogFeature(hog, img.data, img.cols, img.rows, featPtr);
    return;
}

void CorrTrack::getFeatures(Mat &img, Mat &feat, Mat &hannWin, FHOG *hog)
{
    if(feat.data == NULL)
    {
        int rows = getHogFeatureRows(hog, img.rows);
        int cols = getHogFeatureCols(hog, img.cols);
        int channels = getHogFeatureChannels(hog);
        int sz[3] = {channels, rows, cols};
        feat.create(3, sz, CV_32F);
        assert(sz[1] == hannWin.rows && sz[2] == hannWin.cols);
    }
    float *featPtr = feat.ptr<float>(0, 0, 0);
    calcHogFeature(hog, img.data, img.cols, img.rows, featPtr);
    Mat tmp(hog->histHeight, hog->histWidth, CV_32F);
    for(int i = 0; i < hog->dimension; i++)
    {
        featPtr = feat.ptr<float>(i, 0, 0);
        Mat src(hog->histHeight, hog->histWidth, CV_32F, featPtr);
        multiply(src, hannWin, tmp);
        tmp.copyTo(src);
    }
    return;
}

bool CorrTrack::renderHOGFeatures(Mat &feat, Mat &renderImg, FHOG *hog)
{
    if(!hog->glyphs)
        return false;
    if(renderImg.data == NULL)
    {
        int width = hog->glyphSize * hog->histWidth;
        int height = hog->glyphSize * hog->histHeight;
        renderImg.create(height, width, CV_8U);
    }
    float *featPtr = feat.ptr<float>(0, 0, 0);
    renderHogFeature(hog, featPtr, renderImg.data);
    return true;
}

void CorrTrack::fft2(Mat &feat, Mat &featSpectrum)
{
    if(feat.cols == -1 && feat.rows == -1)
    {
        //输入feature为HOG特征
        MatSize sz = feat.size; //尺寸从最高维(层, 第3维)到最低维(列, 第1维)依次排列
        assert(feat.dims == 3);
        assert(isPower2((unsigned int)sz[1]) && isPower2((unsigned int)sz[2]));
        if(featSpectrum.data == NULL)
            featSpectrum.create(3, sz, CV_32FC2);
        for(int i = 0; i < sz[0]; i++)
        {
            float *psrc = feat.ptr<float>(i, 0, 0);
            float *pdst = featSpectrum.ptr<float>(i, 0, 0);
            Mat src(sz[1], sz[2], CV_32F, psrc);
            Mat dst(sz[1], sz[2], CV_32FC2, pdst);
            dft(src, dst, DFT_COMPLEX_OUTPUT);
        }
    }
    else
    {
        //输入feature为RAW特征(灰度特征)
        assert(feat.dims == 2);
        assert(isPower2((unsigned int)feat.cols) && isPower2((unsigned int)feat.rows));
        if(featSpectrum.data == NULL)
            featSpectrum.create(feat.rows, feat.cols, CV_32FC2);
        dft(feat, featSpectrum, DFT_COMPLEX_OUTPUT);
    }
    return;
}

void CorrTrack::ifft2(Mat &spectrum, Mat &response)
{
    if(response.data == NULL)
        response.create(spectrum.rows, spectrum.cols, CV_32F);
    idft(spectrum, response, DFT_SCALE | DFT_REAL_OUTPUT);
    return;
}

void CorrTrack::gaussCorrelationKernel(Mat &xF, Mat &yF, Mat &kernelF, float sigma, bool isTrain)
{
    double xNorm = getCplxNorm(xF);
    double yNorm = isTrain ? xNorm : getCplxNorm(yF);
    Mat xyf;
    if(xF.rows == -1 && xF.cols == -1)
    {
        assert(xF.dims == 3 && yF.dims == 3 && xF.channels() == 2 && xF.channels() == 2);
        MatSize sz = xF.size;
        Mat sum(sz[1], sz[2], CV_32FC2, Scalar::all(0));
        for(int i = 0; i < sz[0]; i++)
        {
            float *pxf = xF.ptr<float>(i, 0, 0);
            float *pyf = yF.ptr<float>(i, 0, 0);
            Mat xtmp(sz[1], sz[2], CV_32FC2, pxf);
            Mat ytmp(sz[1], sz[2], CV_32FC2, pyf);
            Mat tmp(sz[1], sz[2], CV_32FC2);
            mulSpectrums(xtmp, ytmp, tmp, DFT_ROWS, true);
            accumulate(tmp, sum);
        }
        sum.copyTo(xyf);
    }
    else
    {
        assert(xF.dims == 2 && yF.dims == 2 && xF.channels() == 2 && xF.channels() == 2);
        Mat sum(xF.rows, xF.cols, CV_32FC2, Scalar::all(0));
        mulSpectrums(xF, yF, sum, DFT_ROWS, true);
        sum.copyTo(xyf);
    }
    Mat xy(xyf.rows, xyf.cols, CV_32F);
    idft(xyf, xy, DFT_SCALE | DFT_REAL_OUTPUT);
    double scale = -1.0 / (sigma * sigma);
    int n = xy.rows * xy.cols;
    xNorm /= n;
    yNorm /= n;
    for(int i = 0; i < xy.rows; i++)
    {
        float *pxy = xy.ptr<float>(i, 0);
        for(int j = 0; j < xy.cols; j++)
        {
            pxy[j] = MAX_VAL(0, xNorm + yNorm - 2 * pxy[j]);
            pxy[j] *= scale;
            pxy[j] /= n;
            pxy[j] = exp(pxy[j]);
        }
    }
    if(kernelF.data == NULL)
        kernelF.create(xy.rows, xy.cols, CV_32FC2);
    dft(xy, kernelF, DFT_COMPLEX_OUTPUT);
    return;
}

double CorrTrack::getCplxNorm(Mat &src)
{
    assert(src.channels() == 2);
    double value = 0;
    if(src.rows == -1 && src.cols == -1)
    {
        //多维复数矩阵
        assert(src.dims == 3);
        MatSize sz = src.size;
        int nElem = sz[0] * sz[1] * sz[2];
        float *p = src.ptr<float>(0, 0, 0);
        for(int i = 0; i < nElem * 2; i++)
            value += p[i] * p[i];
    }
    else
    {
        //普通的2维复数矩阵
        assert(src.dims == 2);
        int nElem = src.cols * src.rows;
        float *p = src.ptr<float>(0, 0);
        for(int i = 0; i < nElem * 2; i++)
            value += p[i] * p[i];
    }
    return value;
}

void CorrTrack::getSubPixelPeak(Point &maxLoc, Mat &response, Point2f &subPixLoc)
{
    Point p1(maxLoc.x - 1, maxLoc.y - 1);
    Point p2(maxLoc.x, maxLoc.y-1);
    Point p3(maxLoc.x + 1, maxLoc.y - 1);
    Point p4(maxLoc.x - 1, maxLoc.y);
    Point p5(maxLoc.x + 1, maxLoc.y);
    Point p6(maxLoc.x - 1, maxLoc.y + 1);
    Point p7(maxLoc.x, maxLoc.y + 1);
    Point p8(maxLoc.x + 1, maxLoc.y + 1);
    Mat A = (Mat_<float>(9, 6) <<
             p1.x * p1.x, p1.x * p1.y, p1.y * p1.y, p1.x, p1.y, 1.0f,
             p2.x * p2.x, p2.x * p2.y, p2.y * p2.y, p2.x, p2.y, 1.0f,
             p3.x * p3.x, p3.x * p3.y, p3.y * p3.y, p3.x, p3.y, 1.0f,
             p4.x * p4.x, p4.x * p4.y, p4.y * p4.y, p4.x, p4.y, 1.0f,
             p5.x * p5.x, p5.x * p5.y, p5.y * p5.y, p5.x, p5.y, 1.0f,
             p6.x * p6.x, p6.x * p6.y, p6.y * p6.y, p6.x, p6.y, 1.0f,
             p7.x * p7.x, p7.x * p7.y, p7.y * p7.y, p7.x, p7.y, 1.0f,
             p8.x * p8.x, p8.x * p8.y, p8.y * p8.y, p8.x, p8.y, 1.0f,
             maxLoc.x * maxLoc.x, maxLoc.x * maxLoc.y, maxLoc.y * maxLoc.y,
             maxLoc.x, maxLoc.y, 1.0f);
    Mat fval = (Mat_<float>(9, 1) <<
                response.at<float>(p1), response.at<float>(p2), response.at<float>(p3), response.at<float>(p4),
                response.at<float>(p5), response.at<float>(p6), response.at<float>(p7), response.at<float>(p8),
                response.at<float>(maxLoc));
    Mat x;
    solve(A, fval, x, DECOMP_SVD);
    float a = x.at<float>(0);
    float b = x.at<float>(1);
    float c = x.at<float>(2);
    float d = x.at<float>(3);
    float e = x.at<float>(4);
    subPixLoc.y = (2.0f * a * e / b - d) / (b - (4 * a * c) / b);
    subPixLoc.x = (-2.0f * c * subPixLoc.y - e) / b;
    return;
}

void CorrTrack::train(Mat &featSpectrum, Mat &gaussLabelF, Mat &alphaF, float sigma, float lambda)
{
    Mat kernelF;
    gaussCorrelationKernel(featSpectrum, featSpectrum, kernelF, sigma, true);
    if(alphaF.data == NULL)
        alphaF.create(gaussLabelF.rows, gaussLabelF.cols, CV_32FC2);
    for(int i = 0; i < gaussLabelF.rows; i++)
    {
        float *pk = kernelF.ptr<float>(i, 0);
        float *pg = gaussLabelF.ptr<float>(i, 0);
        float *pa = alphaF.ptr<float>(i, 0);
        for(int j = 0; j < gaussLabelF.cols; j++)
        {
            float den = pk[j*2] + lambda;
            pa[j*2] = pg[j*2] / den;
            pa[j*2+1] = pg[j*2+1] / den;
        }
    }
    return;
}

void CorrTrack::detect(Mat &featSpectrum, Mat &featModel, Mat &alphaF, Mat &response, Point2f &pos, float sigma)
{
    Mat kernelF;
    gaussCorrelationKernel(featSpectrum, featModel, kernelF, sigma, false);
    Mat tmp;
    tmp.create(alphaF.rows, alphaF.cols, CV_32FC2);
    mulSpectrums(alphaF, kernelF, tmp, DFT_ROWS, false);
    if(response.data == NULL)
        response.create(alphaF.rows, alphaF.cols, CV_32F);
    ifft2(tmp, response);
    Point maxLoc;
    int maxIdx[2];
    minMaxIdx(response, NULL, NULL, NULL, maxIdx);
    maxLoc.y = maxIdx[0];
    maxLoc.x = maxIdx[1];
    getSubPixelPeak(maxLoc, response, pos);
    return;
}








