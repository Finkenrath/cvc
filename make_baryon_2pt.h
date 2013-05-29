/*****************************************************************
 * make_baryon_2pt.c
 *
 * Mi 8. Mai 07:48:29 EEST 2013
 *
 * PURPOSE
 * - make contraction code for baryon 2-point functions
 *****************************************************************/

extern int perm_tab_3[6][3], perm_tab_3_sign[6], perm_tab_3e[3][3], perm_tab_3o[3][3];
extern int perm_tab_4[24][4], perm_tab_4_sign[24], perm_tab_4e[12][4], perm_tab_4o[12][4], perm_tab_4_sub3[24];

inline void init_perm_tab (void) {
  perm_tab_3[0][0] =  0; 
  perm_tab_3[0][1] =  1; 
  perm_tab_3[0][2] =  2;
  perm_tab_3[1][0] =  0; 
  perm_tab_3[1][1] =  2; 
  perm_tab_3[1][2] =  1;
  perm_tab_3[2][0] =  1; 
  perm_tab_3[2][1] =  0; 
  perm_tab_3[2][2] =  2;
  perm_tab_3[3][0] =  1; 
  perm_tab_3[3][1] =  2; 
  perm_tab_3[3][2] =  0;
  perm_tab_3[4][0] =  2; 
  perm_tab_3[4][1] =  0; 
  perm_tab_3[4][2] =  1;
  perm_tab_3[5][0] =  2; 
  perm_tab_3[5][1] =  1; 
  perm_tab_3[5][2] =  0;
  
  perm_tab_3_sign[0] =  1.;
  perm_tab_3_sign[1] = -1.;
  perm_tab_3_sign[2] = -1.;
  perm_tab_3_sign[3] = 1.;
  perm_tab_3_sign[4] = 1.;
  perm_tab_3_sign[5] = -1.;
  
  /********************************/
  
  perm_tab_3e[0][0] =  0; 
  perm_tab_3e[0][1] =  1; 
  perm_tab_3e[0][2] =  2;
  
  perm_tab_3e[1][0] =  1; 
  perm_tab_3e[1][1] =  2; 
  perm_tab_3e[1][2] =  0;
  
  perm_tab_3e[2][0] =  2; 
  perm_tab_3e[2][1] =  0; 
  perm_tab_3e[2][2] =  1;
  
  perm_tab_3o[0][0] =  0; 
  perm_tab_3o[0][1] =  2; 
  perm_tab_3o[0][2] =  1;
  
  perm_tab_3o[1][0] =  2; 
  perm_tab_3o[1][1] =  1; 
  perm_tab_3o[1][2] =  0;
  
  perm_tab_3o[2][0] =  1; 
  perm_tab_3o[2][1] =  0; 
  perm_tab_3o[2][2] =  2;

/********************************/

  perm_tab_4_sign[ 0] = +1;
  perm_tab_4_sign[ 1] = -1;
  perm_tab_4_sign[ 2] = +1;
  perm_tab_4_sign[ 3] = -1;
  perm_tab_4_sign[ 4] = +1;
  perm_tab_4_sign[ 5] = -1;
  perm_tab_4_sign[ 6] = -1;
  perm_tab_4_sign[ 7] = +1;
  perm_tab_4_sign[ 8] = -1;
  perm_tab_4_sign[ 9] = +1;
  perm_tab_4_sign[10] = -1;
  perm_tab_4_sign[11] = +1;
  perm_tab_4_sign[12] = +1;
  perm_tab_4_sign[13] = +1;
  perm_tab_4_sign[14] = -1;
  perm_tab_4_sign[15] = -1;
  perm_tab_4_sign[16] = -1;
  perm_tab_4_sign[17] = -1;
  perm_tab_4_sign[18] = +1;
  perm_tab_4_sign[19] = +1;
  perm_tab_4_sign[20] = -1;
  perm_tab_4_sign[21] = +1;
  perm_tab_4_sign[22] = +1;
  perm_tab_4_sign[23] = -1;

  perm_tab_4[0][0] = 0;
  perm_tab_4[0][1] = 1;
  perm_tab_4[0][2] = 2;
  perm_tab_4[0][3] = 3;
  
  perm_tab_4[1][0] = 1;
  perm_tab_4[1][1] = 2;
  perm_tab_4[1][2] = 3;
  perm_tab_4[1][3] = 0;
  
  perm_tab_4[2][0] = 2;
  perm_tab_4[2][1] = 3;
  perm_tab_4[2][2] = 0;
  perm_tab_4[2][3] = 1;
  
  perm_tab_4[3][0] = 3;
  perm_tab_4[3][1] = 0;
  perm_tab_4[3][2] = 1;
  perm_tab_4[3][3] = 2;
  
  perm_tab_4[4][0] = 0;
  perm_tab_4[4][1] = 2;
  perm_tab_4[4][2] = 3;
  perm_tab_4[4][3] = 1;
  
  perm_tab_4[5][0] = 1;
  perm_tab_4[5][1] = 0;
  perm_tab_4[5][2] = 2;
  perm_tab_4[5][3] = 3;
  
  perm_tab_4[6][0] = 2;
  perm_tab_4[6][1] = 3;
  perm_tab_4[6][2] = 1;
  perm_tab_4[6][3] = 0;
  
  perm_tab_4[7][0] = 3;
  perm_tab_4[7][1] = 1;
  perm_tab_4[7][2] = 0;
  perm_tab_4[7][3] = 2;
  
  perm_tab_4[8][0] = 0;
  perm_tab_4[8][1] = 3;
  perm_tab_4[8][2] = 2;
  perm_tab_4[8][3] = 1;
  
  perm_tab_4[9][0] = 1;
  perm_tab_4[9][1] = 0;
  perm_tab_4[9][2] = 3;
  perm_tab_4[9][3] = 2;
  
  perm_tab_4[10][0] = 2;
  perm_tab_4[10][1] = 1;
  perm_tab_4[10][2] = 0;
  perm_tab_4[10][3] = 3;
  
  perm_tab_4[11][0] = 3;
  perm_tab_4[11][1] = 2;
  perm_tab_4[11][2] = 1;
  perm_tab_4[11][3] = 0;
  
  perm_tab_4[12][0] = 0;
  perm_tab_4[12][1] = 3;
  perm_tab_4[12][2] = 1;
  perm_tab_4[12][3] = 2;
  
  perm_tab_4[13][0] = 1;
  perm_tab_4[13][1] = 2;
  perm_tab_4[13][2] = 0;
  perm_tab_4[13][3] = 3;
  
  perm_tab_4[14][0] = 2;
  perm_tab_4[14][1] = 0;
  perm_tab_4[14][2] = 3;
  perm_tab_4[14][3] = 1;
  
  perm_tab_4[15][0] = 3;
  perm_tab_4[15][1] = 1;
  perm_tab_4[15][2] = 2;
  perm_tab_4[15][3] = 0;
  
  perm_tab_4[16][0] = 0;
  perm_tab_4[16][1] = 2;
  perm_tab_4[16][2] = 1;
  perm_tab_4[16][3] = 3;
  
  perm_tab_4[17][0] = 1;
  perm_tab_4[17][1] = 3;
  perm_tab_4[17][2] = 0;
  perm_tab_4[17][3] = 2;
  
  perm_tab_4[18][0] = 2;
  perm_tab_4[18][1] = 1;
  perm_tab_4[18][2] = 3;
  perm_tab_4[18][3] = 0;
  
  perm_tab_4[19][0] = 3;
  perm_tab_4[19][1] = 0;
  perm_tab_4[19][2] = 2;
  perm_tab_4[19][3] = 1;
  
  perm_tab_4[20][0] = 0;
  perm_tab_4[20][1] = 1;
  perm_tab_4[20][2] = 3;
  perm_tab_4[20][3] = 2;
  
  perm_tab_4[21][0] = 1;
  perm_tab_4[21][1] = 3;
  perm_tab_4[21][2] = 2;
  perm_tab_4[21][3] = 0;
  
  perm_tab_4[22][0] = 2;
  perm_tab_4[22][1] = 0;
  perm_tab_4[22][2] = 1;
  perm_tab_4[22][3] = 3;
  
  perm_tab_4[23][0] = 3;
  perm_tab_4[23][1] = 2;
  perm_tab_4[23][2] = 0;
  perm_tab_4[23][3] = 1;
  
  /********************************/
  
  perm_tab_4e[0][0] = 0;
  perm_tab_4e[0][1] = 1;
  perm_tab_4e[0][2] = 2;
  perm_tab_4e[0][3] = 3;
  
  perm_tab_4e[1][0] = 0;
  perm_tab_4e[1][1] = 2;
  perm_tab_4e[1][2] = 3;
  perm_tab_4e[1][3] = 1;
  
  perm_tab_4e[2][0] = 0;
  perm_tab_4e[2][1] = 3;
  perm_tab_4e[2][2] = 1;
  perm_tab_4e[2][3] = 2;
  
  perm_tab_4e[3][0] = 1;
  perm_tab_4e[3][1] = 0;
  perm_tab_4e[3][2] = 3;
  perm_tab_4e[3][3] = 2;
  
  perm_tab_4e[4][0] = 1;
  perm_tab_4e[4][1] = 2;
  perm_tab_4e[4][2] = 0;
  perm_tab_4e[4][3] = 3;
  
  perm_tab_4e[5][0] = 1;
  perm_tab_4e[5][1] = 3;
  perm_tab_4e[5][2] = 2;
  perm_tab_4e[5][3] = 0;
  
  perm_tab_4e[6][0] = 2;
  perm_tab_4e[6][1] = 0;
  perm_tab_4e[6][2] = 1;
  perm_tab_4e[6][3] = 3;
  
  perm_tab_4e[7][0] = 2;
  perm_tab_4e[7][1] = 1;
  perm_tab_4e[7][2] = 3;
  perm_tab_4e[7][3] = 0;
  
  perm_tab_4e[8][0] = 2;
  perm_tab_4e[8][1] = 3;
  perm_tab_4e[8][2] = 0;
  perm_tab_4e[8][3] = 1;
  
  perm_tab_4e[9][0] = 3;
  perm_tab_4e[9][1] = 0;
  perm_tab_4e[9][2] = 2;
  perm_tab_4e[9][3] = 1;
  
  perm_tab_4e[10][0] = 3;
  perm_tab_4e[10][1] = 1;
  perm_tab_4e[10][2] = 0;
  perm_tab_4e[10][3] = 2;
  
  perm_tab_4e[11][0] = 3;
  perm_tab_4e[11][1] = 2;
  perm_tab_4e[11][2] = 1;
  perm_tab_4e[11][3] = 0;
  
  /********************************/
  
  perm_tab_4o[0][0] = 0;
  perm_tab_4o[0][1] = 2;
  perm_tab_4o[0][2] = 1;
  perm_tab_4o[0][3] = 3;
  
  perm_tab_4o[1][0] = 0;
  perm_tab_4o[1][1] = 3;
  perm_tab_4o[1][2] = 2;
  perm_tab_4o[1][3] = 1;
  
  perm_tab_4o[2][0] = 0;
  perm_tab_4o[2][1] = 1;
  perm_tab_4o[2][2] = 3;
  perm_tab_4o[2][3] = 2;
  
  perm_tab_4o[3][0] = 1;
  perm_tab_4o[3][1] = 3;
  perm_tab_4o[3][2] = 0;
  perm_tab_4o[3][3] = 2;
  
  perm_tab_4o[4][0] = 1;
  perm_tab_4o[4][1] = 0;
  perm_tab_4o[4][2] = 2;
  perm_tab_4o[4][3] = 3;
  
  perm_tab_4o[5][0] = 1;
  perm_tab_4o[5][1] = 2;
  perm_tab_4o[5][2] = 3;
  perm_tab_4o[5][3] = 0;
  
  perm_tab_4o[6][0] = 2;
  perm_tab_4o[6][1] = 1;
  perm_tab_4o[6][2] = 0;
  perm_tab_4o[6][3] = 3;
  
  perm_tab_4o[7][0] = 2;
  perm_tab_4o[7][1] = 3;
  perm_tab_4o[7][2] = 1;
  perm_tab_4o[7][3] = 0;
  
  perm_tab_4o[8][0] = 2;
  perm_tab_4o[8][1] = 0;
  perm_tab_4o[8][2] = 3;
  perm_tab_4o[8][3] = 1;
  
  perm_tab_4o[9][0] = 3;
  perm_tab_4o[9][1] = 2;
  perm_tab_4o[9][2] = 0;
  perm_tab_4o[9][3] = 1;
  
  perm_tab_4o[10][0] = 3;
  perm_tab_4o[10][1] = 0;
  perm_tab_4o[10][2] = 1;
  perm_tab_4o[10][3] = 2;
  
  perm_tab_4o[11][0] = 3;
  perm_tab_4o[11][1] = 1;
  perm_tab_4o[11][2] = 2;
  perm_tab_4o[11][3] = 0;

  perm_tab_4_sub3_sign[ 0] = +1;
  perm_tab_4_sub3_sign[ 1] = +1;
  perm_tab_4_sub3_sign[ 2] = +1;
  perm_tab_4_sub3_sign[ 3] = +1;
  perm_tab_4_sub3_sign[ 4] = +1;
  perm_tab_4_sub3_sign[ 5] = +1;
  perm_tab_4_sub3_sign[ 6] = +1;
  perm_tab_4_sub3_sign[ 7] = +1;
  perm_tab_4_sub3_sign[ 8] = -1;
  perm_tab_4_sub3_sign[ 9] = -1;
  perm_tab_4_sub3_sign[10] = -1;
  perm_tab_4_sub3_sign[11] = -1;
  perm_tab_4_sub3_sign[12] = +1;
  perm_tab_4_sub3_sign[13] = +1;
  perm_tab_4_sub3_sign[14] = +1;
  perm_tab_4_sub3_sign[15] = +1;
  perm_tab_4_sub3_sign[16] = -1;
  perm_tab_4_sub3_sign[17] = -1;
  perm_tab_4_sub3_sign[18] = -1;
  perm_tab_4_sub3_sign[19] = -1;
  perm_tab_4_sub3_sign[20] = -1;
  perm_tab_4_sub3_sign[21] = -1;
  perm_tab_4_sub3_sign[22] = -1;
  perm_tab_4_sub3_sign[23] = -1;
}  // end of init_perm_tab

typedef struct baryon_current_operator_struct {
  int n;
  int **flavor;
  int *Gamma;
  double *c;
} baryon_current_operator;

typedef struct baryon_current_correlator_struct {
  int n;
  int **flavor;
  int **Gamma;
  double *c;
  int *permid;
} baryon_current_correlator;

typedef struct meson_current_operator_struct {
  int n;
  int **flavor;
  int *Gamma;
  double *c;
} meson_current_operator;

void init_baryon_current_operator (baryon_current_operator*j);

void fini_baryon_current_operator (baryon_current_operator*j);

void printf_baryon_current_operator (baryon_current_operator*j, FILE*ofs);

void alloc_baryon_current_operator (baryon_current_operator*j, int n);

void read_baryon_current_operator (baryon_current_operator*j, FILE*ifs);

void adjoint_baryon_current_operator (baryon_current_operator*jad, baryon_current_operator*j);

void init_baryon_current_correlator (baryon_current_correlator*s);

void fini_baryon_current_correlator (baryon_current_correlator*s);

void alloc_baryon_current_correlator (baryon_current_correlator*s, int n);

void contract_baryon_current_operator (baryon_current_correlator*s, baryon_current_operator*o1, baryon_current_operator*o2);

void printf_baryon_current_correlator (baryon_current_correlator*s, FILE*ofs);
