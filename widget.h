#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QImage>
#include <QStringList>
#include <QList>
#include <QTimer>
#include <QLabel>
#include <QEvent>
#include <QPoint>

#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include "corrtrack.h"

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();

private:
    Ui::Widget *ui;
    TrackParam *param;
    CorrTrack *fsct;
    QTimer *timerPlay;
    QTimer *timerTrack;
    QTime *time;
    QImage frame;
    QImage curApp;
    QImage glbApp;

    QStringList picSeq;
    QList<cv::Rect> groundTruth;
    cv::VideoCapture video;
    cv::Mat frameBuf;

    cv::Rect targetRect;
    QRect rect;
    int frameNum;
    int frameCount;
    bool pushButtonInitState;
    bool pushButtonTrackingState;
    bool pushButtonSetParamState;
    bool isInitState;
    bool isTrackingState;
    bool hasInitialized;
    bool m_isDown;
    QPoint m_start;
    QPoint m_stop;
    QPointF frameOrigin;

private slots:
    void getParamFromUi();
    void showPlayFrame();
    void showTrackingFrame();
    void onPushButtonBrowse_clicked();
    void onPushButtonReset_clicked();
    void onPushButtonInit_clicked();
    void onPushButtonTracking_clicked();
    void onComboBoxSource_currentIndexChanged(int index);


private:
    void setDefaultParam();
    void setUiParamProperty();
    void setInputWidgetAvaliable(bool avaliable);
    void listPicFiles();
    void readGroundTruth();
    void fillFrameBuf();

    void openInputSource();

    void initFristFrame();
    void tracking();

    void mousePressEvent(QMouseEvent *e);
    void mouseMoveEvent(QMouseEvent *e);
    void mouseReleaseEvent(QMouseEvent *e);
    void paintEvent(QPaintEvent *);
    void showTrackingImage(const cv::Mat& mat, QImage& frame, QLabel *label);
    void cvRect2QRectF(const cv::Rect& cvrect, QRectF &qrect, double zoom);
    void showPlayImage(const cv::Mat& mat, QImage& frame, QLabel *label);
    QImage cvMat2QImage(const cv::Mat& mat);
};

#endif // WIDGET_H
