/*****************************************************************
 * make_baryon_2pt.c
 *
 * Mi 8. Mai 07:48:29 EEST 2013
 *
 * PURPOSE
 * - make contraction code for baryon 2-point functions
 *****************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

int adjoint_current_gamma_sign[] = {-1,+1,+1,+1,-1,+1,-1,+1,+1,+1,-1,-1,-1,+1,+1,+1};

int perm_tab_3[6][3], perm_tab_3_sign[6], perm_tab_3e[3][3], perm_tab_3o[3][3];
int perm_tab_4[24][4], perm_tab_4_sign[24], perm_tab_4e[12][4], perm_tab_4o[12][4], perm_tab_4_sub3[24];

/*************************************************************************************************************
 * functions for baryon operators
 *************************************************************************************************************/

void init_baryon_current_operator (baryon_current_operator*j) {
  j->n      = 0;
  j->flavor = NULL;
  j->Gamma  = NULL;
  j->c      = NULL;
}

void fini_baryon_current_operator (baryon_current_operator*j) {
  j->n      = 0;
  if(j->flavor != NULL) {
    if(j->flavor[0]!=NULL) { free(j->flavor[0]);}
    free(j->flavor);
    j->flavor = NULL;
  }
  if(j->Gamma != NULL) {
    free(j->Gamma);
    j->Gamma = NULL;
  }
  if(j->c != NULL) {
    free(j->c);
    j->c = NULL;
  }
}

void printf_baryon_current_operator (baryon_current_operator*j, FILE*ofs) {

  int i;
  fprintf(ofs, "# [printf_baryon_current_operator] number of summands = %d\n", j->n);
  for(i=0; i<j->n; i++) {
    fprintf(ofs, "\t%3d\t(%2d,%2d,%2d)\t%3d\t%16.7e\n", i, j->flavor[i][0], j->flavor[i][1], j->flavor[i][2],
       j->Gamma[i], j->c[i]);
  }
}

void alloc_baryon_current_operator (baryon_current_operator*j, int n) {
  int i;
  j->n = n;
  j->flavor = (int**)malloc(j->n*sizeof(int*));
  j->flavor[0] = (int*)malloc(3*j->n*sizeof(int));
  for(i=1; i<j->n; i++) j->flavor[i] = j->flavor[i-1] + 3;

  j->Gamma = (int*)malloc(j->n * sizeof(int));
  j->c = (double*)malloc(j->n * sizeof(double));
}

void read_baryon_current_operator (baryon_current_operator*j, FILE*ifs) {
  int n;
  char line[400];

  n=0;
  while(fgets(line, 400, ifs) != NULL) {
    if(line[0]=='#') continue;
    n++;
  }
  fprintf(stdout, "# [read_baryon_current_operator] found %d operator terms\n", n);
  if(n==0) return;
  alloc_baryon_current_operator (j, n);

  rewind(ifs);
  n=0;
  while(fgets(line, 400, ifs) != NULL) {
    if(line[0]=='#') continue;
    sscanf(line, "%d%d%d%d%lf\n", j->flavor[n], j->flavor[n]+1, j->flavor[n]+2, j->Gamma+n, j->c+n);
    n++;
  }
}

void adjoint_baryon_current_operator (baryon_current_operator*jad, baryon_current_operator*j) {
  int i;
  init_baryon_current_operator(jad);
  alloc_baryon_current_operator(jad, j->n);
  memcpy(jad->c, j->c, j->n * sizeof(double));
  memcpy(jad->Gamma, j->Gamma, j->n * sizeof(int));
  for(i=0; i<jad->n; i++) {
    jad->flavor[i][0] = j->flavor[i][2];
    jad->flavor[i][1] = j->flavor[i][1];
    jad->flavor[i][2] = j->flavor[i][0];
    jad->c[i] *= adjoint_current_gamma_sign[jad->Gamma[i]];
  }
}

/*************************************************************************************************************
 * functions for baryon correlators
 *************************************************************************************************************/

void init_baryon_current_correlator (baryon_current_correlator*s) {
  s->n      = 0;
  s->flavor = NULL;
  s->Gamma  = NULL;
  s->c      = NULL;
  s->permid = NULL;
}

void fini_baryon_current_correlator (baryon_current_correlator*s) {
  s->n      = 0;
  if(s->flavor != NULL) {
    if(s->flavor[0] != NULL) free(s->flavor[0]);
    free(s->flavor);
    s->flavor = NULL;
  }
  if(s->Gamma != NULL) {
    if(s->Gamma[0] != NULL) free(s->Gamma[0]);
    free(s->Gamma);
    s->Gamma = NULL;
  }
  if(s->c != NULL) {
    free(s->c);
    s->c = NULL;
  }
  if(s->permid != NULL) {
    free(s->permid);
    s->permid   = NULL;
  }
}

void alloc_baryon_current_correlator (baryon_current_correlator*s, int n) {
  int i;
  s->n      = n;
  s->flavor = (int**)malloc(s->n*sizeof(int*));
  s->flavor[0] = (int*)malloc(3*s->n*sizeof(int));
  for(i=1; i<s->n; i++) s->flavor[i] = s->flavor[i-1] + 3;
  s->c      = (double*)malloc(s->n * sizeof(double));
  s->permid   = (int*)malloc(s->n * sizeof(int));
  s->Gamma  = (int**)malloc(s->n*sizeof(int*));
  s->Gamma[0] = (int*)malloc(2*s->n*sizeof(int)); 
  for(i=1; i<s->n; i++) s->Gamma[i] = s->Gamma[i-1] + 2;
}

void contract_baryon_current_operator (baryon_current_correlator*s, baryon_current_operator*o1, baryon_current_operator*o2) {
  int n1, n2, n12_max, count, iperm, have_contrib;

  n12_max = o1->n * o2->n * 6;
  fprintf(stdout, "# [contract_baryon_current_operator] maximal number of contributions = %3d\n", n12_max);

  alloc_baryon_current_correlator(s, n12_max);

  count = 0;
  for(n1=0; n1<o1->n; n1++) {
  for(n2=0; n2<o2->n; n2++) {
    for(iperm=0; iperm<6; iperm++) {
      // fprintf(stdout, "# [contract_baryon_current_operator] checking (%3d,%3d,%3d) (%3d,%3d,%3d) %3d\n",
          o1->flavor[n1][0], o1->flavor[n1][1], o1->flavor[n1][2], 
          o2->flavor[n2][0], o2->flavor[n2][1], o2->flavor[n2][2],
          iperm);
      have_contrib = ( o1->flavor[n1][0] == o2->flavor[n2][perm_tab_3[iperm][2]] )
                  && ( o1->flavor[n1][1] == o2->flavor[n2][perm_tab_3[iperm][1]] )
                  && ( o1->flavor[n1][2] == o2->flavor[n2][perm_tab_3[iperm][0]] );
      if(!have_contrib) continue;
      fprintf(stdout, "# [contract_baryon_current_operator] found contribution no. %d, (%3d,%3d) %3d\n", count+1, n1, n2, iperm);

      memcpy(s->flavor[count], o1->flavor[n1], 3*sizeof(int));
      fprintf(stdout, "# [contract_baryon_current_operator] flavor f = (%3d,%3d,%3d)\n",
          s->flavor[count][0], s->flavor[count][1], s->flavor[count][2]);
      s->c[count] = o1->c[n1] * o2->c[n2];
      s->Gamma[count][0] = o1->Gamma[n1];
      s->Gamma[count][1] = o2->Gamma[n2];
      s->permid[count] = iperm;
      count++;
    }
  }}

  s->n = count;
}

void printf_baryon_current_correlator (baryon_current_correlator*s, FILE*ofs) {
  int n;

  for(n=0; n<s->n; n++) {
    fprintf(ofs, "S_{f%d}^{i%d j%d} ", s->flavor[n][0], 0, perm_tab_3[s->permid[n]][2]);
    fprintf(ofs, "S_{f%d}^{i%d j%d} ", s->flavor[n][1], 1, perm_tab_3[s->permid[n]][1]);
    fprintf(ofs, "S_{f%d}^{i%d j%d} ", s->flavor[n][2], 2, perm_tab_3[s->permid[n]][0]);
    fprintf(ofs, "Gamma%.2d_{i0 i1} Gamma%.2d_{j1 j2} %16.7e\n", s->Gamma[n][0], s->Gamma[n][1], s->c[n]);
  }
}

/*************************************************************************************************************
 * functions for meson operators
 *************************************************************************************************************/

void init_meson_current_operator (meson_current_operator*j) {
  j->n      = 0;
  j->flavor = NULL;
  j->Gamma  = NULL;
  j->c      = NULL;
}

void fini_meson_current_operator (meson_current_operator*j) {
  j->n      = 0;
  if(j->flavor != NULL) {
    if(j->flavor[0]!=NULL) { free(j->flavor[0]);}
    free(j->flavor);
    j->flavor = NULL;
  }
  if(j->Gamma != NULL) {
    free(j->Gamma);
    j->Gamma = NULL;
  }
  if(j->c != NULL) {
    free(j->c);
    j->c = NULL;
  }
}

void alloc_meson_current_operator (meson_current_operator*j, int n) {
  int i;
  j->n = n;
  j->flavor = (int**)malloc(j->n*sizeof(int*));
  j->flavor[0] = (int*)malloc(2*j->n*sizeof(int));
  for(i=1; i<j->n; i++) j->flavor[i] = j->flavor[i-1] + 2;

  j->Gamma = (int*)malloc(j->n * sizeof(int));
  j->c = (double*)malloc(j->n * sizeof(double));
}

void printf_meson_current_operator (meson_current_operator*j, FILE*ofs) {

  int i;
  fprintf(ofs, "# [printf_meson_current_operator] number of summands = %d\n", j->n);
  for(i=0; i<j->n; i++) {
    fprintf(ofs, "\t%3d\t(%2d,%2d)\t%3d\t%16.7e\n", i, j->flavor[i][0], j->flavor[i][1],
        j->Gamma[i], j->c[i]);
  }
}

void read_meson_current_operator (meson_current_operator*j, FILE*ifs) {
  int n;
  char line[400];

  n=0;
  while(fgets(line, 400, ifs) != NULL) {
    if(line[0]=='#') continue;
    n++;
  }
  fprintf(stdout, "# [read_meson_current_operator] found %d operator terms\n", n);
  if(n==0) return;
  alloc_meson_current_operator (j, n);

  rewind(ifs);
  n=0;
  while(fgets(line, 400, ifs) != NULL) {
    if(line[0]=='#') continue;
    sscanf(line, "%d%d%d%lf\n", j->flavor[n], j->flavor[n]+1, j->Gamma+n, j->c+n);
    n++;
  }
}


#define MAIN_PROGRAM

/*************************************************************************************************************
 * main program for tests
 *************************************************************************************************************/

int main(int argc, char **argv) {
  int c;
  int verbose = 0;
  char input_filename[200];
  int input_filename_set=0;
  baryon_current_operator bco, bco2;
  baryon_current_correlator bcc;
  FILE *ifs=NULL;

  while ((c = getopt(argc, argv, "h?vi:")) != -1) {
    switch (c) {
      case 'v':
        verbose = 1;
        break;
      case 'i':
        strcpy(input_filename, optarg);
        input_filename_set = 1;
        fprintf(stdout, "# [] will read input from file %s\n", input_filename);
        break;
      case '?':
      case 'h':
      default:
        fprintf(stderr, "[] Error, unrecognized option\n");
        exit(1);
        break;
    }
  }

  if(!input_filename_set) {
    fprintf(stderr, "[] Error, input filename was not set\n");
    exit(2);
  }

  init_perm_tab ();

  init_baryon_current_operator (&bco);

  ifs = fopen(input_filename, "r"); 
  read_baryon_current_operator (&bco, ifs);
  fclose(ifs); ifs = NULL;
  printf_baryon_current_operator (&bco, stdout);
  adjoint_baryon_current_operator (&bco2, &bco);
  printf_baryon_current_operator (&bco2, stdout);

  init_baryon_current_correlator (&bcc);
  contract_baryon_current_operator (&bcc, &bco, &bco2);
  printf_baryon_current_correlator (&bcc, stdout);


  fini_baryon_current_operator (&bco);
  fini_baryon_current_operator (&bco2);
  fini_baryon_current_correlator (&bcc);
  return(0);
}
