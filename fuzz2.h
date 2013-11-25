#ifndef _FUZZ2_H
#define _FUZZ2_H


int Fuzz_prop3(double *fuzzed_gauge_field, double *psi, double *psi_old, const int Nlong);
int fuzzed_links2(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong);
#endif
