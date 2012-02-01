#ifndef _DQUANT_H
#define _DQUANT_H

/* pointer to function of type f(nalpha, alpha, npara, para) */
typedef int (*dquant)(int, double*, int, double*, double*);

#endif
