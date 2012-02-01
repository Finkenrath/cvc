#ifndef _MAKE_H3ORBITS_H
#  define _MAKE_H3ORBITS_H

# define _sqr(_x) ((_x)*(_x))
# define _qrt(_x) (_sqr(_x)*_sqr(_x))
# define _hex(_x) (_qrt(_x)*_sqr(_x))
# define _oct(_x) (_qrt(_x)*_qrt(_x))

void set_qid_val(double **qid_val, int ts);
int make_H3orbits(int **qid, int **qid_count, double ***qid_val, int *nc);
int make_H4orbits(int **qid, int **qid_count, double ***qid_val, int *nc);
int make_H3orbits_timeslice(int **qid, int **qid_count, double ***qid_val, int *nc);
int make_Oh_orbits_r(int **rid, int **rid_count, double ***rid_val, int *nc, const double Rmin, const double Rmax);

int make_H4_r_orbits(int **qid, int **qid_count, double ***qid_val, int *nc, int***rep);
void finalize_H4_r_orbits(int **qid, int **qid_count, double ***qid_val, int***rep);

void init_perm_tabs(void);

#endif
