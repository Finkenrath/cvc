#ifndef _PIDISC_MODEL_H

#define _PIDISC_MODEL_H

#ifdef MPI
#  define _FFTW_PLAN fftwnd_mpi_plan
#else
#  define _FFTW_PLAN fftwnd_plan
#endif

#if defined MAIN_PROGRAM
#  define EXTERN
#else
#  define EXTERN extern
#endif

EXTERN int (*model_type_function)(const double, const double, const double, double*, _FFTW_PLAN, const int);

int pidisc_model(const double mrho, const double dcoeffre, const double dcoeffim,double *pimn, _FFTW_PLAN plan_m, const int imu);
int pidisc_model1(const double mrho, const double dcoeffre, const double dcoeffim,double *pimn, _FFTW_PLAN plan_m, const int imu);
int pidisc_model2(const double mrho, const double dcoeffre, const double dcoeffim,double *pimn, _FFTW_PLAN plan_m, const int imu);
int pidisc_model3(const double mrho, const double dcoeffre, const double dcoeffim,double *pimn, _FFTW_PLAN plan_m, const int imu);

#endif

