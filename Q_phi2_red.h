#ifndef Q_PHI2_ITER_H
#define Q_PHI2_ITER_H
void reduce_loop_tab(int **loop_tab, int **sigma_tab, int **shift_start, int deg, int nloop);
void Hopping_iter_red(double *truf, double *trub, double *tcf, double *tcb, int xd,
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start);
void init_trace_coeff_red(double **tcf, double **tcb, int ***loop_tab, int ***sigma_tab, int ***shift_start, int deg, int *N, int mudir);
void init_lvc_trace_coeff_red(double **tcf, int ***loop_tab, int ***sigma_tab, int ***shift_start, int deg, int *N, int mudir);
void Hopping_lvc_iter_red(double *truf, double *tcf, int xd,
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start);
void Hopping_iter_mc_red(double *truf, double *tcf, int xd,
  int mu, int deg, int nloop, int **loop_tab, int **sigma_tab, int **shift_start);

#endif
