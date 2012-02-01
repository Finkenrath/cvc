#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H
int prepare_timeslice_source(double *s, double *gauge_field, int timeslice, unsigned int V, char*rng_file_in, char*rng_file_out);
int prepare_coherent_timeslice_source(double *s, double *gauge_field, int base, int delta, unsigned int V, char*rng_file_in, char*rng_file_out);
int prepare_timeslice_source_one_end(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset);
int prepare_timeslice_source_one_end_color(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset);
#endif
