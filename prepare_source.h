#ifndef _PREPARE_SOURCE_H
#define _PREPARE_SOURCE_H
int prepare_timeslice_source(double *s, double *gauge_field, int timeslice, unsigned int V, int*rng_state, int rng_reset);
int prepare_timeslice_source_one_end(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset);
int prepare_timeslice_source_one_end_color(double *s, double *gauge_field, int timeslice, int*momentum, unsigned int isc, int*rng_state, int rng_reset);
int prepare_coherent_timeslice_source(double *s, double *gauge_field, int base, int delta, unsigned int V, int*rng_state, int rng_reset);
int prepare_sequential_point_source (double*source, int isc, int timeslice, int*momentum, int smear, double*work, double*gauge_field_smeared);
int prepare_space_diluted_source(double *s, unsigned int degree, unsigned int number, int is, int ic, int*isLat, int*rng_state, int rng_reset);
#endif
