#ifndef _MAKE_CUTLIST_H
#  define _MAKE_CUTLIST_H

# define _sqr(_x) ((_x)*(_x))
# define _qrt(_x) (_sqr(_x)*_sqr(_x))
# define _hex(_x) (_qrt(_x)*_sqr(_x))
# define _oct(_x) (_qrt(_x)*_qrt(_x))

int make_cutid_list(int *idlist, int *cutdir, double rad, double angle);
int make_cutid_list2(int *idlist, int *cutdir, double rad, double angle);
#endif
