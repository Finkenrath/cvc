#ifndef _MAKE_X_ORBITS_H
#define _MAKE_X_ORBITS_H
int make_x_orbits_3d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep);
void finalize_x_orbits(int **xid, int **xid_count, double ***xid_val, int***xid_rep);
void finalize_x_orbits2(int **xid, int **xid_count, double ***xid_val, int***xid_rep, int****xid_member);
int init_x_orbits(int**xid, int**xid_count, double ***xid_val, int***xid_rep, int Nclasses);
int make_x_orbits_4d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ****xid_member);
int reduce_x_orbits_4d(int *xid, int *xid_count, double **xid_val, int xid_nc, int **xid_rep, int ***xid_member);
int make_x_orbits_4d_symmetric(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep, int ****xid_member);
#endif
