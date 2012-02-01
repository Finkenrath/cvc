#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "incomp_gamma.h"

double incomp_gamma(
		 double U  /* upper limit of integration, in */,
		 int Nexp  /* exponent in the integrand */) {

  int  i;
  double norm;

  if(Nexp==1) {
    return(1-exp(-U));
  }
  else {
    for(i=2, norm=1.0; i<Nexp; norm=norm * ((double)i++));
    return(-pow(U,Nexp-1)*exp(-U)/norm + incomp_gamma(U, Nexp-1));
  }
}


/*   T = Tlast = 0.0; */
/*   S = Slast = 0.0; */
/*   N = 1; */
/*   h1 = U; */
/*   h  = U; */
/*   for(count=0; count<MAX_STEP; count++) { */
/*     if(count==0) { */
/*       T = h/2.0*(Func(0.0,Nexp) + Func(U,Nexp)); */
/*     } */
/*     else { */
/*       h1 = h1 / 2.0; */
/*       Tlast = T; */
/*       tnew = 0.0; */
/*       for(i=0; i<N; i++) { */
/* 	tnew = tnew + Func(h1+(double)i*h, Nexp); */
/*       } */
/*       T =0.5 * (T + h*tnew); */
/*       Slast = S; */
/*       S = (4.0*T-Tlast)/3.0; */
/*       if((S-Slast>=0 && S-Slast<prec) || (S-Slast<0.0 && Slast-S <prec)) break; */
/*       N = 2*N; */
/*       h  = h  / 2.0; */
/*     } */
/*   } */
/*   if(count<MAX_STEP) { */
/*     fprintf(stdout, "[incomp_gamma]: precision reached after %d steps\n", count); */
/*     *P = S; */
/*   } */
/*   else { */
/*     fprintf(stdout, "[incomp_gamma]: precision not reached after %d steps\n", count); */
/*     *P = 0.0; */
/*     return(1); */
/*   } */
