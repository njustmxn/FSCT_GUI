#include "widget.h"
#include "ui_widget.h"
#include "corrtrack.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <QString>
#include <QStringList>
#include <QTextStream>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QDir>
#include <QFile>
#include <QPixmap>
#include <QImage>
#include <QPainter>
#include <QRect>
#include <QLabel>
#include <QTimer>
#include <QTime>
#include <QMouseEvent>
#include <QLCDNumber>

#include <QDebug>

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    param = new TrackParam;
    setDefaultParam();
    setUiParamProperty();
    QObject::connect(ui->spinBoxTransPattSz, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->spinBoxTransCellSz, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->spinBoxTransGaussSigmaRate, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->spinBoxScalePattSz, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->spinBoxScaleCellSz, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->spinBoxScaleGaussSigmaRate, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->doubleSpinBoxTransPad, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->doubleSpinBoxTransLearnRate, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->doubleSpinBoxScaleMinRhoCoef, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->doubleSpinBoxScaleLearnRate, SIGNAL(editingFinished()), this, SLOT(getParamFromUi()));
    QObject::connect(ui->checkBoxUseScale, SIGNAL(stateChanged(int)), this, SLOT(getParamFromUi()));
    QObject::connect(ui->comboBoxSource, SIGNAL(currentIndexChanged(int)), this, SLOT(onComboBoxSource_currentIndexChanged(int)));
    QObject::connect(ui->pushButtonBrowse, SIGNAL(clicked()), this, SLOT(onPushButtonBrowse_clicked()));

    timerPlay = new QTimer;
    timerPlay->stop();
    QObject::connect(timerPlay, SIGNAL(timeout()), this, SLOT(showPlayFrame()));

    pushButtonSetParamState = false;
    ui->pushButtonInit->setEnabled(false);
    ui->pushButtonTracking->setEnabled(false);
    ui->pushButtonSetParam->setEnabled(true);
    QObject::connect(ui->pushButtonSetParam, SIGNAL(clicked()), this, SLOT(onPushButtonReset_clicked()));
    QObject::connect(ui->pushButtonInit, SIGNAL(clicked()), this, SLOT(onPushButtonInit_clicked()));
    QObject::connect(ui->pushButtonTracking, SIGNAL(clicked()), this, SLOT(onPushButtonTracking_clicked()));

    timerTrack = new QTimer;
    timerTrack->stop();
    QObject::connect(timerTrack, SIGNAL(timeout()), this, SLOT(showTrackingFrame()));

    QPalette lcdpat;
    lcdpat.setColor(QPalette::Normal, QPalette::WindowText, Qt::black);
    ui->lcdNumberFrame->setMode(QLCDNumber::Dec);
    ui->lcdNumberFrame->setDigitCount(4);
    ui->lcdNumberFrame->setSegmentStyle(QLCDNumber::Flat);
    ui->lcdNumberFrame->setPalette(lcdpat);
    ui->lcdNumberFps->setMode(QLCDNumber::Dec);
    ui->lcdNumberFps->setDigitCount(4);
    ui->lcdNumberFps->setSegmentStyle(QLCDNumber::Flat);
    ui->lcdNumberFps->setPalette(lcdpat);

    time = new QTime;
    fsct = new CorrTrack;

}

Widget::~Widget()
{
    delete timerTrack;
    delete timerPlay;
    delete time;
    delete fsct;
    delete param;
    delete ui;
}

void Widget::onPushButtonBrowse_clicked()
{
    QString filename;
    if(ui->comboBoxSource->currentIndex() == 1)
        filename = QFileDialog::getOpenFileName(this, "Open Video",
                                                QDir::currentPath(),
                                                "Videos (*.avi)");
    else if(ui->comboBoxSource->currentIndex() == 2)
        filename = QFileDialog::getExistingDirectory(this, "Open Image Sequence Directory",
                                                     QDir::currentPath(),
                                                     QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    ui->lineEditPath->setText(filename);
}

void Widget::showPlayFrame()
{
    if(frameNum < frameCount)
    {
        fillFrameBuf();
        showPlayImage(frameBuf, frame, ui->labelFrame);
        frameNum++;
        ui->lcdNumberFrame->display(frameNum-1);
        timerPlay->start(20);
    }
}

void Widget::showTrackingFrame()
{
    if(frameNum < frameCount)
    {
        time->start();
        fillFrameBuf();

        fsct->trackEachFrame(frameBuf, targetRect);

        showTrackingImage(frameBuf, frame, ui->labelFrame);
        showPlayImage(fsct->globalApp, glbApp, ui->labelGlbApp);
        showPlayImage(fsct->currentApp, curApp, ui->labelCurApp);
        int fps = 1000 / time->elapsed();
        frameNum++;
        ui->lcdNumberFrame->display(frameNum-1);
        ui->lcdNumberFps->display(fps);
        timerTrack->start(1);
    }
}

void Widget::setDefaultParam()
{
    param->transPad = 1.5;
    param->transPattSz = 32;
    param->transCellSz = 4;
    param->transGaussSigmaRate = 24;
    param->transLearnRate = 0.02;
    param->useScale = true;
    param->scaleMinRhoCoef = 0.2;
    param->scalePattSz = 32;
    param->scaleCellSz = 4;
    param->scaleGaussSigmaRate = 28;
    param->scaleLearnRate = 0.02;
}

void Widget::getParamFromUi()
{
    param->transPad = ui->doubleSpinBoxTransPad->value();
    param->transPattSz = ui->spinBoxTransPattSz->value();
    param->transCellSz = ui->spinBoxTransCellSz->value();
    param->transGaussSigmaRate = ui->spinBoxTransGaussSigmaRate->value();
    param->transLearnRate = ui->doubleSpinBoxTransLearnRate->value();
    param->useScale = ui->checkBoxUseScale->isChecked();
    param->scaleMinRhoCoef = ui->doubleSpinBoxScaleMinRhoCoef->value();
    param->scalePattSz = ui->spinBoxScalePattSz->value();
    param->scaleCellSz = ui->spinBoxScaleCellSz->value();
    param->scaleGaussSigmaRate = ui->spinBoxScaleGaussSigmaRate->value();
    param->scaleLearnRate = ui->doubleSpinBoxScaleLearnRate->value();
}

void Widget::setUiParamProperty()
{
    ui->doubleSpinBoxTransPad->setValue(param->transPad);
    ui->doubleSpinBoxTransPad->setRange(0.5, 2.0);
    ui->doubleSpinBoxTransPad->setSingleStep(0.05);
    ui->spinBoxTransPattSz->setValue(param->transPattSz);
    ui->spinBoxTransPattSz->setRange(16, 128);
    ui->spinBoxTransPattSz->setSingleStep(4);
    ui->spinBoxTransCellSz->setValue(param->transCellSz);
    ui->spinBoxTransCellSz->setRange(2, 8);
    ui->spinBoxTransCellSz->setSingleStep(2);
    ui->spinBoxTransGaussSigmaRate->setValue(param->transGaussSigmaRate);
    ui->spinBoxTransGaussSigmaRate->setRange(8, 128);
    ui->spinBoxTransGaussSigmaRate->setSingleStep(1);
    ui->doubleSpinBoxTransLearnRate->setValue(param->transLearnRate);
    ui->doubleSpinBoxTransLearnRate->setRange(0.001, 1.000);
    ui->doubleSpinBoxTransLearnRate->setSingleStep(0.001);
    ui->checkBoxUseScale->setChecked(param->useScale);
    ui->doubleSpinBoxScaleMinRhoCoef->setValue(param->scaleMinRhoCoef);
    ui->doubleSpinBoxScaleMinRhoCoef->setRange(0.01, 1.00);
    ui->doubleSpinBoxScaleMinRhoCoef->setSingleStep(0.01);
    ui->spinBoxScalePattSz->setValue(param->scalePattSz);
    ui->spinBoxScalePattSz->setRange(16, 128);
    ui->spinBoxScalePattSz->setSingleStep(4);
    ui->spinBoxScaleCellSz->setValue(param->scaleCellSz);
    ui->spinBoxScaleCellSz->setRange(2, 8);
    ui->spinBoxScaleCellSz->setSingleStep(2);
    ui->spinBoxScaleGaussSigmaRate->setValue(param->scaleGaussSigmaRate);
    ui->spinBoxScaleGaussSigmaRate->setRange(8, 128);
    ui->spinBoxScaleGaussSigmaRate->setSingleStep(1);
    ui->doubleSpinBoxScaleLearnRate->setValue(param->scaleLearnRate);
    ui->doubleSpinBoxScaleLearnRate->setRange(0.001, 1.000);
    ui->doubleSpinBoxScaleLearnRate->setSingleStep(0.001);
}

void Widget::setInputWidgetAvaliable(bool avaliable)
{
    ui->checkBoxInitManu->setEnabled(avaliable);
    ui->checkBoxUseScale->setEnabled(avaliable);
    ui->comboBoxSource->setEnabled(avaliable);
    ui->spinBoxScaleCellSz->setEnabled(avaliable);
    ui->spinBoxScaleGaussSigmaRate->setEnabled(avaliable);
    ui->spinBoxScalePattSz->setEnabled(avaliable);
    ui->spinBoxTransCellSz->setEnabled(avaliable);
    ui->spinBoxTransGaussSigmaRate->setEnabled(avaliable);
    ui->spinBoxTransPattSz->setEnabled(avaliable);
    ui->doubleSpinBoxScaleLearnRate->setEnabled(avaliable);
    ui->doubleSpinBoxScaleMinRhoCoef->setEnabled(avaliable);
    ui->doubleSpinBoxTransLearnRate->setEnabled(avaliable);
    ui->doubleSpinBoxTransPad->setEnabled(avaliable);
    ui->pushButtonBrowse->setEnabled(avaliable);
    ui->lineEditPath->setEnabled(avaliable);
}

void Widget::fillFrameBuf()
{
    if(ui->comboBoxSource->currentIndex() != 2)
        video >> frameBuf;
    else
        frameBuf = cv::imread(picSeq.at(frameNum).toStdString());
}

void Widget::openInputSource()
{
    switch(ui->comboBoxSource->currentIndex())
    {
    case 0:
        video.open(0);
        if(!video.isOpened())
            qDebug() << "Can not initialize camera!";
        frameCount = 9999;
        break;
    case 1:
        video.open(ui->lineEditPath->text().toStdString());
        if(!video.isOpened())
            qDebug() << "Can not open video file!";
        frameCount = (int)video.get(cv::CAP_PROP_FRAME_COUNT) - 1;
        break;
    case 2:
        if(!picSeq.isEmpty())
            picSeq.clear();
        if(!groundTruth.isEmpty())
            groundTruth.clear();
        listPicFiles();
        if(!ui->checkBoxInitManu->isChecked())
            readGroundTruth();
        frameCount = picSeq.length() - 1;
        break;
    }
}

void Widget::listPicFiles()
{
    QString path = ui->lineEditPath->text();
    if(path[path.length()-1] != '/')
        path += '/';
    path += "img/";
    QDir dir(path);
    if(! dir.exists())
        qDebug() << path << " is not a valid path!";
    QStringList filters;
    filters << "*.jpg" << ".bmp";
    QStringList files = dir.entryList(filters, QDir::Files, QDir::Name);
    for(int i = 0; i < files.length(); i++)
        picSeq.push_back(path + files.at(i));
}

void Widget::readGroundTruth()
{
    QString path = ui->lineEditPath->text();
    if(path[path.length()-1] != '/')
        path += '/';
    path += "groundtruth_rect.txt";
    QFile file(path);
    if(!file.exists())
        qDebug() << "groundtruth_rect.txt does not exist!";
    file.open(QIODevice::ReadOnly);
    QTextStream in(&file);
    cv::Rect tmp;
    while(!in.atEnd())
    {
        QString line = in.readLine();
        QStringList strlist;
        if(line.contains(","))
        {
            strlist = line.split(",", QString::SkipEmptyParts);
            tmp.x = strlist.at(0).toInt();
            tmp.y = strlist.at(1).toInt();
            tmp.width = strlist.at(2).toInt();
            tmp.height = strlist.at(3).toInt();
            groundTruth.push_back(tmp);
        }
        else
        {
            strlist = line.split(QRegExp("\\s+"));
            tmp.x = strlist.at(0).toInt();
            tmp.y = strlist.at(1).toInt();
            tmp.width = strlist.at(2).toInt();
            tmp.height = strlist.at(3).toInt();
            groundTruth.push_back(tmp);
        }
    }
}

QImage Widget::cvMat2QImage(const cv::Mat& mat)
{
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
            image.setColor(i, qRgb(i, i, i));
        uchar *psrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pdst = image.scanLine(row);
            memcpy(pdst, psrc, mat.cols);
            psrc += mat.step;
        }
        return image;
    }
    else if(mat.type() == CV_8UC3)
    {
        const uchar *psrc = (const uchar*)mat.data;
        QImage image(psrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else
    {
        qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

void Widget::showPlayImage(const cv::Mat &mat, QImage &frame, QLabel *label)
{
    QImage img = cvMat2QImage(mat);
    if(mat.rows <= label->height() && mat.cols <= label->width())
    {
        frame = img;
        label->setPixmap(QPixmap::fromImage(img));
    }
    else if((mat.rows > label->height() || mat.cols > label->width())
            && mat.rows >= mat.cols)
    {
        frame = img.scaledToHeight(label->height());
        label->setPixmap(QPixmap::fromImage(frame));
    }
    else if((mat.rows > label->height() || mat.cols > label->width())
            && mat.rows < mat.cols)
    {
        frame = img.scaledToWidth(label->width());
        label->setPixmap(QPixmap::fromImage(frame));
    }
    label->setAlignment(Qt::AlignCenter);
}

void Widget::showTrackingImage(const cv::Mat &mat, QImage &frame, QLabel *label)
{
    QImage img = cvMat2QImage(mat);
    QImage scaleImg;
    double zoom;
    if(mat.rows <= label->height() && mat.cols <= label->width())
    {
        scaleImg = img;
        zoom = 1;
    }
    else if((mat.rows > label->height() || mat.cols > label->width())
            && mat.rows >= mat.cols)
    {
        scaleImg = img.scaledToHeight(label->height());
        zoom = label->height() / mat.rows;
    }
    else if((mat.rows > label->height() || mat.cols > label->width())
            && mat.rows < mat.cols)
    {
        scaleImg = img.scaledToWidth(label->width());
        zoom = label->width() / mat.cols;
    }
//    QImage convertImg = scaleImg.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QPainter painter(&frame);
    painter.setCompositionMode(QPainter::CompositionMode_Source);
    painter.drawImage(0, 0, scaleImg);
    QRectF gt;
    if(!groundTruth.isEmpty())
    {
        cvRect2QRectF(groundTruth.at(frameNum), gt, zoom);
        painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
        painter.setPen(QPen(QBrush(Qt::green), 2.0));
        painter.drawRect(gt);
    }
    QRectF tgt;
    cvRect2QRectF(targetRect, tgt, zoom);
    painter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    painter.setPen(QPen(QBrush(Qt::red), 2.0));
    painter.drawRect(tgt);
    painter.end();
    label->setPixmap(QPixmap::fromImage(frame));
    label->setAlignment(Qt::AlignCenter);
}

void Widget::cvRect2QRectF(const cv::Rect &cvrect, QRectF &qrect, double zoom)
{
    qrect.setX(cvrect.x * zoom);
    qrect.setY(cvrect.y * zoom);
    qrect.setWidth(cvrect.width * zoom);
    qrect.setHeight(cvrect.height * zoom);
}

void Widget::onPushButtonReset_clicked()
{
    pushButtonSetParamState = !pushButtonSetParamState; //改变（设置/重置）按钮状态
    if(pushButtonSetParamState) //Set功能
    {
        ui->pushButtonSetParam->setText("Reset Parameter"); //改变（设置/重置）按钮文字
        ui->pushButtonInit->setEnabled(true); //设置（播放/初始化）按钮可用
        ui->pushButtonInit->setText("Play"); //重新设置（播放/初始化）按钮文字
        setInputWidgetAvaliable(false); //设置参数输入框不可用
        fsct->initParam(param);
        openInputSource();
    }
    else //Reset功能
    {
        ui->pushButtonSetParam->setText("Set Parameter"); //改变（设置/重置）按钮文字
        ui->pushButtonInit->setEnabled(false); //设置（播放/初始化）按钮不可用
        setInputWidgetAvaliable(true); //设置参数输入框不可用
        setDefaultParam(); //重置默认参数
        setUiParamProperty(); //设置UI显示参数
        groundTruth.clear();
        picSeq.clear();
        if(video.isOpened())
            video.release();
    }
    pushButtonInitState = false; //重新设置（播放/初始化）按钮功能
    pushButtonTrackingState = false; //重新设置（跟踪）按钮功能
    isInitState = false;
    isTrackingState = false;
    hasInitialized = false;
    ui->pushButtonTracking->setText("Tracking"); //重新设置（跟踪）按钮文字
    ui->pushButtonTracking->setEnabled(false); //设置（跟踪）按钮不可用
    frameNum = 0;    
    m_isDown = false;
    m_start = QPoint(0,0);
    m_stop = QPoint(0,0);
    pushButtonInitState = false;
    pushButtonTrackingState = false;
    timerPlay->stop();
    timerTrack->stop();
    ui->lcdNumberFrame->display(0);
    ui->lcdNumberFps->display(0);
    ui->labelFrame->clear();
    ui->labelGlbApp->clear();
    ui->labelCurApp->clear();
}

void Widget::onPushButtonInit_clicked()
{
    pushButtonInitState = !pushButtonInitState; //切换状态
    if(pushButtonInitState)
    {
        isInitState = false; //首先是播放功能，此时关闭初始化画框功能
        ui->pushButtonInit->setText("Initialize"); //改变按钮文字，下一次点击为初始化功能
        ui->pushButtonTracking->setEnabled(false); //设置Tracking按钮不可用
        timerPlay->start(5); //启动定时器，开始播放
    }
    else
    {
        isInitState = true; //开启初始化目标的画框功能
        ui->pushButtonInit->setText("Play"); //改变按钮文字，下一次点击为继续播放功能
        ui->pushButtonTracking->setEnabled(true); //设置Tracking按钮可用
        timerPlay->stop(); //暂停定时器，开始初始化画框功能
    }
}

void Widget::onPushButtonTracking_clicked()
{
    pushButtonTrackingState = !pushButtonTrackingState;
    isInitState = false;
    ui->pushButtonInit->setEnabled(false); //禁用播放按钮
    if(!hasInitialized)
    {
        if(!ui->checkBoxInitManu->isChecked() && ui->comboBoxSource->currentIndex() == 2)
        {
            targetRect = groundTruth.at(frameNum-1);
        }
        else
        {
            targetRect.x = qMin(m_start.x(), m_stop.x()) - frameOrigin.x();
            targetRect.y = qMin(m_start.y(), m_stop.y()) - frameOrigin.y();
            targetRect.width = qAbs(m_stop.x() - m_start.x()) + 1;
            targetRect.height = qAbs(m_stop.y() - m_start.y()) + 1;
        }
        fsct->initTarget(frameBuf, targetRect);
        hasInitialized = true;
    }
    if(pushButtonTrackingState) //跟踪状态
    {
        isTrackingState = true;
        ui->pushButtonTracking->setText("Stop");
        timerTrack->start(1);
    }
    else //暂停状态
    {
        isTrackingState = false;
        ui->pushButtonTracking->setText("Continue");
        timerTrack->stop();
    }
}

void Widget::onComboBoxSource_currentIndexChanged(int index)
{
    if(index) //如果输入源为外部视频或图像序列，则启用Browse按钮和路径输入框
    {
        ui->lineEditPath->setEnabled(true);
        ui->lineEditPath->clear();
        ui->pushButtonBrowse->setEnabled(true);
    }
    else //如果输入源为外部视频或图像序列，则禁用Browse按钮和路径输入框
    {
        ui->lineEditPath->setEnabled(false);
        ui->lineEditPath->clear();
        ui->pushButtonBrowse->setEnabled(false);
    }
}

void Widget::mousePressEvent(QMouseEvent *e)
{
    if(e->button() && Qt::LeftButton && isInitState)
    {
        m_isDown = true;
        m_start = e->pos();
        m_stop = e->pos();
    }
}

void Widget::mouseMoveEvent(QMouseEvent *e)
{
    if(m_isDown && isInitState)
        m_stop = e->pos();
    ui->labelFrame->update(); //只更新labelFrame范围内的绘图事件
}

void Widget::mouseReleaseEvent(QMouseEvent *e)
{
    if(e->button() && Qt::LeftButton && isInitState)
        m_isDown = false;
}

void Widget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    if(!m_isDown && !isInitState)
    {
        p.end();
    }
    else
    {
        ui->labelFrame->clear();
        QRect pg(ui->labelFrame->geometry());
        double xOffset = (ui->labelFrame->contentsRect().width() - frame.width()) / 2;
        double yOffset = (ui->labelFrame->contentsRect().height() - frame.height()) / 2;
        QPointF leftup(pg.x() + xOffset, pg.y() + yOffset);
        frameOrigin = leftup.toPoint();
        p.drawImage(leftup, frame);
        p.setPen(QPen(QBrush(Qt::red), 2.0));
        p.drawRect(QRect(m_start,m_stop));
    }
}


