#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <getopt.h>

#define MAIN_PROGRAM

int main (int argc, char **argv) {

  int L = 4;
  int Lhalf = L/2;
  int Lhp1 = L/2 + 1;
  int Lhm1 = L/2 -1;
  int x0, x1, x2, x3;

  int Nclasses = + (Lhalf+1) * (Lhalf) * (Lhalf-1) * 3 / 12;

  int count = 0;
  for(x0=0; x0<Lhp1-2; x0++) {
  for(x1=x0+1; x1<Lhp1-1; x1++) {
  for(x2=x1+1; x2<Lhp1; x2++) {
    printf("# x[%4d] = %2d  %2d  %2d\n", count/3, x0, x1, x2);
  for(x3=0; x3<3; x3++) {
    count++;
  }}}}

  printf("# Nclasses:\n");
  printf("\tcalculated : %d\n", Nclasses);
  printf("\tcounted    : %d\n", count);
  return(0);

}

