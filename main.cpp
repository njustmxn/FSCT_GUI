#include "widget.h"
#include <QApplication>

#ifdef STATIC_PROG
#pragma comment( lib, "vfw32.lib" )
#pragma comment( lib, "comctl32.lib" )
#endif

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Widget w;
    w.show();

    return a.exec();
}
