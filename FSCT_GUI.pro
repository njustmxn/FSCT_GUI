#-------------------------------------------------
#
# Project created by QtCreator 2016-12-15T22:17:51
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FSCT_GUI
TEMPLATE = app

CONFIG(debug, debug|release) {
    DESTDIR =       $$PWD/debug
    OBJECTS_DIR =   $$PWD/debug/obj
    MOC_DIR =       $$PWD/debug/moc
    RCC_DIR =       $$PWD/debug/rcc
    UI_DIR =        $$PWD/debug/ui
}

CONFIG(release, debug|release) {
    DESTDIR =       $$PWD/release
    OBJECTS_DIR =   $$PWD/release/obj
    MOC_DIR =       $$PWD/release/moc
    RCC_DIR =       $$PWD/release/rcc
    UI_DIR =        $$PWD/release/ui
}

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

INCLUDEPATH += D:\OpenCV3.1.0\build\include

CONFIG(release, debug|release){
LIBS += -LD:\OpenCV3.1.0\build\x64\vc10\lib \
    -lopencv_world310 \

}
CONFIG(debug, debug|release){
LIBS += -LD:\OpenCV3.1.0\build\x64\vc10\lib \
    -lopencv_world310d
}

RC_FILE += icon.rc
RESOURCES += res.qrc

