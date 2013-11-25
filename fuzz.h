#ifndef _FUZZ_H
#define _FUZZ_H

int fuzzed_links_Timeslice(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong, const int timeslice);

int Fuzz_prop(double *fuzzed_gauge_field, double *psi, const int Nlong);
int Fuzz_prop2(double *fuzzed_gauge_field, double *psi, double *psi_old, const int Nlong);
int fuzzed_links(double *fuzzed_gauge_field, double *smeared_gauge_field, const int Nlong);
#endif
