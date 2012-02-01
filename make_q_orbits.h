#ifndef _MAKE_Q_ORBITS_H
#define _MAKE_Q_ORBITS_H
void finalize_q_orbits(int **xid, int **xid_count, double ***xid_val, int***xid_rep);
int init_q_orbits(int**xid, int**xid_count, double ***xid_val, int***xid_rep, int Nclasses);
int make_q_orbits_3d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, double qmax);
int make_qlatt_orbits_3d_parity_avg(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ***xmap);
int make_qcont_orbits_3d_parity_avg(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ***xmap);
#endif
