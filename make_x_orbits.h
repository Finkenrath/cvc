#ifndef _MAKE_X_ORBITS_H
#define _MAKE_X_ORBITS_H
int make_x_orbits_3d(int **xid, int **xid_count, double ***xid_val, int *xid_nc, int ***xid_rep);
void finalize_x_orbits(int **xid, int **xid_count, double ***xid_val, int***xid_rep);
int init_x_orbits(int**xid, int**xid_count, double ***xid_val, int***xid_rep, int Nclasses);

#endif
