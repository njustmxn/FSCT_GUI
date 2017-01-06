#-------------------------------------------------
#
# Project created by QtCreator 2016-12-15T22:17:51
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FSCT_GUI
TEMPLATE = app


SOURCES += main.cpp\
        widget.cpp \
    corrtrack.cpp \
    hog.c \
    lpt.c

HEADERS  += widget.h \
    corrtrack.h \
    hog.h \
    lpt.h

FORMS    += widget.ui

INCLUDEPATH += D:\OpenCV3.1\build\include \
                D:\OpenCV3.1\build\include\opencv \
                D:\OpenCV3.1\build\include\opencv2

#CONFIG(release, debug|release){
#LIBS += -LD:\OpenCV3.1\build\x86\vc10\lib\ \
#    -lopencv_world310 \

#}
#CONFIG(debug, debug|release){
#LIBS += -LD:\OpenCV3.1\build\x86\vc10\lib\ \
#    -lopencv_world310d
#}

RC_FILE += icon.rc

DEFINES += STATIC_PROG
Release:LIBS += -LD:\OpenCV3.1\build\x86\vc10\staticlib\ \
    -lopencv_world310 \
    -lopencv_ts310 \
    -lippicvmt \
    -llibjpeg \
    -lzlib \
    -llibwebp \
    -lIlmImf \
    -llibjasper

RESOURCES += \
    res.qrc
