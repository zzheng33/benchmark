#ifndef TIMER_H
#define TIMER_H

#include <ctime>

class Timer {
public:
  Timer();

  void start();
  void stop();
  void print();
  double elapse();

private:
  clock_t t1, t2;
  int flag;
};

#endif
