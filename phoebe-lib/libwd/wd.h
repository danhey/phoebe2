#ifndef WD_H
	#define WD_H 1

#include <f2c.h>

extern int atmx_(doublereal *t, doublereal *g, integer *ifil, doublereal *xintlog, doublereal *xint);
extern int bbl_(doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *fr1, doublereal *fr2, doublereal *hld, doublereal *slump1, doublereal *slump2, doublereal *theta, doublereal *rho, doublereal *aa, doublereal *bb, doublereal *phsv, doublereal *pcsv, integer *n1, integer *n2, doublereal *f1, doublereal *f2, doublereal *d__, doublereal *hlum, doublereal *clum, doublereal *xh, doublereal *xc, doublereal *yh, doublereal *yc, doublereal *gr1, doublereal *gr2, doublereal *wl, doublereal *sm1, doublereal *sm2, doublereal *tpolh, doublereal *tpolc, doublereal *sbrh, doublereal *sbrc, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, doublereal *phas, doublereal *rm, doublereal *xincl, doublereal *hot, doublereal *cool, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *tld, doublereal *glump1, doublereal *glump2, doublereal *xx1, doublereal *xx2, doublereal *yy1, doublereal *yy2, doublereal *zz1, doublereal *zz2, doublereal *dint1, doublereal *dint2, doublereal *grv1, doublereal *grv2, doublereal *rftemp, doublereal *rf1, doublereal *rf2, doublereal *csbt1, doublereal *csbt2, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, doublereal *fbin1, doublereal *fbin2, doublereal *delv1, doublereal *delv2, doublereal *count1, doublereal *count2, doublereal *delwl1, doublereal *delwl2, doublereal *resf1, doublereal *resf2, doublereal *wl1, doublereal *wl2, doublereal *dvks1, doublereal *dvks2, doublereal *tau1, doublereal *tau2, doublereal *emm1, doublereal *emm2, doublereal *hbarw1, doublereal *hbarw2, doublereal *xcl, doublereal *ycl, doublereal *zcl, doublereal *rcl, doublereal *op1, doublereal *fcl, doublereal *dens, doublereal *encl, doublereal *edens, doublereal *taug, doublereal *emmg, doublereal *yskp, doublereal *zskp, integer *mode, integer *iband, integer *ifat1, integer *ifat2, integer *ifphn);
extern int binnum_(doublereal *x, integer *n, doublereal *y, integer *j);
extern int cloud_(doublereal *cosa, doublereal *cosb, doublereal *cosg, doublereal *x1, doublereal *y1, doublereal *z1, doublereal *xc, doublereal *yc, doublereal *zc, doublereal *rr, doublereal *wl, doublereal *op1, doublereal *opsf, doublereal *edens, doublereal *acm, doublereal *en, doublereal *cmpd, doublereal *ri, doublereal *dx, doublereal *dens, doublereal *tau);
extern int conjph_(doublereal *ecc, doublereal *argper, doublereal *phzero, doublereal *trsc, doublereal *tric, doublereal *econsc, doublereal *econic, doublereal *xmsc, doublereal *xmic, doublereal *pconsc, doublereal *pconic);
extern int dc_(char *atmtab, char *pltab, integer *l3perc, doublereal *corrs, doublereal *stdevs, doublereal *chi2s, doublereal *cfval, ftnlen atmtab_len, ftnlen pltab_len);
extern int dgmprd_(doublereal *a, doublereal *b, doublereal *r__, integer *n, integer *m, integer *l);
extern int dminv_(doublereal *a, integer *n, doublereal *d__, integer *l, integer *m);
extern int dura_(doublereal *f, doublereal *xincl, doublereal *rm, doublereal *d__, doublereal *the, doublereal *omeg, doublereal *r__);
extern int ellone_(doublereal *ff, doublereal *dd, doublereal *rm, doublereal *xl1, doublereal *om1, doublereal *xl2, doublereal *om2);
extern int fourls_(doublereal *th, doublereal *ro, integer *nobs, integer *nth, doublereal *aa, doublereal *bb);
extern int gabs_(integer *komp, doublereal *smaxis, doublereal *qq, doublereal *ecc, doublereal *period, doublereal *dd, doublereal *rad, doublereal *xm, doublereal *xmo, doublereal *absgr, doublereal *glog);
extern int jdph_(doublereal *xjdin, doublereal *phin, doublereal *t0, doublereal *p0, doublereal *dpdt, doublereal *xjdout, doublereal *phout);
extern int kepler_(doublereal *xm, doublereal *ec, doublereal *ecan, doublereal *tr);
extern int lcr_(doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *fr1, doublereal *fr2, doublereal *hld, doublereal *slump1, doublereal *slump2, doublereal *rm, doublereal *poth, doublereal *potc, integer *n1, integer *n2, doublereal *f1, doublereal *f2, doublereal *d__, doublereal *hlum, doublereal *clum, doublereal *xh, doublereal *xc, doublereal *yh, doublereal *yc, doublereal *gr1, doublereal *gr2, doublereal *sm1, doublereal *sm2, doublereal *tpolh, doublereal *tpolc, doublereal *sbrh, doublereal *sbrc, integer *ifat1, integer *ifat2, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, doublereal *vol1, doublereal *vol2, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *tld, doublereal *glump1, doublereal *glump2, doublereal *xx1, doublereal *xx2, doublereal *yy1, doublereal *yy2, doublereal *zz1, doublereal *zz2, doublereal *dint1, doublereal *dint2, doublereal *grv1, doublereal *grv2, doublereal *csbt1, doublereal *csbt2, doublereal *rftemp, doublereal *rf1, doublereal *rf2, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, integer *mode, integer *iband);
extern int lc_(char *atmtab, char *pltab, integer *request, integer *vertno, integer *l3perc, doublereal *indeps, doublereal *deps, doublereal *skycoy, doublereal *skycoz, doublereal *params, ftnlen atmtab_len, ftnlen pltab_len);
extern int legendre_(doublereal *x, doublereal *pleg, integer *n);
extern int light_(doublereal *phs, doublereal *xincl, doublereal *xh, doublereal *xc, doublereal *yh, doublereal *yc, integer *n1, integer *n2, doublereal *sumhot, doublereal *sumkul, doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *theta, doublereal *rho, doublereal *aa, doublereal *bb, doublereal *slump1, doublereal *slump2, doublereal *somhot, doublereal *somkul, doublereal *d__, doublereal *wl, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *tld, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, doublereal *fbin1, doublereal *fbin2, doublereal *delv1, doublereal *delv2, doublereal *count1, doublereal *count2, doublereal *delwl1, doublereal *delwl2, doublereal *resf1, doublereal *resf2, doublereal *wl1, doublereal *wl2, doublereal *dvks1, doublereal *dvks2, doublereal *tau1, doublereal *tau2, doublereal *emm1, doublereal *emm2, doublereal *hbarw1, doublereal *hbarw2, doublereal *xcl, doublereal *ycl, doublereal *zcl, doublereal *rcl, doublereal *op1, doublereal *fcl, doublereal *edens, doublereal *encl, doublereal *dens, doublereal *taug, doublereal *emmg, doublereal *yskp, doublereal *zskp, integer *iband, integer *ifat1, integer *ifat2, integer *ifphn);
extern int linpro_(integer *komp, doublereal *dvks, doublereal *hbarw, doublereal *tau, doublereal *emm, doublereal *count, doublereal *taug, doublereal *emmg, doublereal *fbin, doublereal *delv);
extern int lum_(doublereal *xlum, doublereal *x, doublereal *y, doublereal *tpoll, integer *n, integer *n1, integer *komp, doublereal *sbr, doublereal *rv, doublereal *rvq, doublereal *glump1, doublereal *glump2, doublereal *glog1, doublereal *glog2, doublereal *grv1, doublereal *grv2, integer *mmsave, doublereal *summ, doublereal *fr, doublereal *sm, integer *ifat, doublereal *vol, doublereal *rm, doublereal *om, doublereal *f, doublereal *d__, doublereal *snth, integer *iband);
extern int lump_(doublereal *grx, doublereal *gry, doublereal *grz, doublereal *grxq, doublereal *gryq, doublereal *grzq, doublereal *slump1, doublereal *slump2, integer *mmsave, doublereal *alb, doublereal *tpoll, doublereal *sbr, integer *n1, integer *n2, integer *komp, integer *ifat, doublereal *fr, doublereal *snth, doublereal *tld, doublereal *glump1, doublereal *glump2, doublereal *xx1, doublereal *xx2, doublereal *yy1, doublereal *yy2, doublereal *zz1, doublereal *zz2, doublereal *xbol, doublereal *ybol, doublereal *grv1, doublereal *grv2, doublereal *sbr1b, doublereal *sbr2b, doublereal *rf, doublereal *rfo, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, doublereal *dint, integer *iband);
extern int mlrg_(doublereal *a, doublereal *p, doublereal *q, doublereal *r1, doublereal *r2, doublereal *t1, doublereal *t2, doublereal *sm1, doublereal *sm2, doublereal *sr1, doublereal *sr2, doublereal *bolm1, doublereal *bolm2, doublereal *xlg1, doublereal *xlg2);
extern int modlog_(doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *fr1, doublereal *fr2, doublereal *hld, doublereal *rm, doublereal *poth, doublereal *potc, doublereal *gr1, doublereal *gr2, doublereal *alb1, doublereal *alb2, integer *n1, integer *n2, doublereal *f1, doublereal *f2, integer *mod, doublereal *xincl, doublereal *the, integer *mode, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *grv1, doublereal *grv2, doublereal *xx1, doublereal *yy1, doublereal *zz1, doublereal *xx2, doublereal *yy2, doublereal *zz2, doublereal *glump1, doublereal *glump2, doublereal *csbt1, doublereal *csbt2, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2);
extern int nekmin_(doublereal *rm, doublereal *omeg, doublereal *x, doublereal *z__);
extern int olump_(doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, doublereal *slump1, doublereal *slump2, integer *mmsave, doublereal *grexp, doublereal *alb, doublereal *rb, doublereal *tpoll, doublereal *sbr, doublereal *summ, integer *n1, integer *n2, integer *komp, integer *ifat, doublereal *x, doublereal *y, doublereal *d__, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *tld, doublereal *glump1, doublereal *glump2, doublereal *glog1, doublereal *glog2, doublereal *grv1, doublereal *grv2, integer *iband);
extern int planckint_(doublereal *t, integer *ifil, doublereal *ylog, doublereal *y);
extern int rangau_(doublereal *smod, integer *nn, doublereal *sd, doublereal *gau);
extern int ranuni_(doublereal *sn, doublereal *smod, doublereal *sm1p1);
extern int rddata_(integer *mpage, integer *nref, integer *mref, integer *ifsmv1, integer *ifsmv2, integer *icor1, integer *icor2, integer *ld, integer *jdphs, doublereal *hjd0, doublereal *period, doublereal *dpdt, doublereal *pshift, doublereal *stdev, integer *noise, doublereal *seed, doublereal *hjdst, doublereal *hjdsp, doublereal *hjdin, doublereal *phstrt, doublereal *phstop, doublereal *phin, doublereal *phn, integer *mode, integer *ipb, integer *ifat1, integer *ifat2, integer *n1, integer *n2, doublereal *perr0, doublereal *dperdt, doublereal *the, doublereal *vunit, doublereal *e, doublereal *a, doublereal *f1, doublereal *f2, doublereal *vga, doublereal *xincl, doublereal *gr1, doublereal *gr2, doublereal *abunin, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *poth, doublereal *potc, doublereal *rm, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, integer *iband, doublereal *hlum, doublereal *clum, doublereal *xh, doublereal *xc, doublereal *yh, doublereal *yc, doublereal *el3, doublereal *opsf, doublereal *zero, doublereal *factor, doublereal *wl, doublereal *binwm1, doublereal *sc1, doublereal *sl1, doublereal *wll1, doublereal *ewid1, doublereal *depth1, integer *kks, doublereal *binwm2, doublereal *sc2, doublereal *sl2, doublereal *wll2, doublereal *ewid2, doublereal *depth2, doublereal *xlat, doublereal *xlong, doublereal *radsp, doublereal *temsp, doublereal *xcl, doublereal *ycl, doublereal *zcl, doublereal *rcl, doublereal *op1, doublereal *fcl, doublereal *edens, doublereal *xmue, doublereal *encl, integer *lpimax, integer *ispmax, integer *iclmax);
extern int ring_(doublereal *q, doublereal *om, integer *komp, integer *l, doublereal *fr, doublereal *hld, doublereal *r1, doublereal *rl);
extern int romq_(doublereal *omein, doublereal *q, doublereal *f, doublereal *d__, doublereal *ec, doublereal *th, doublereal *fi, doublereal *r__, doublereal *drdo, doublereal *drdq, doublereal *dodq, integer *komp, integer *mode);
extern int sincos_(integer *komp, integer *n, integer *n1, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, integer *mmsave);
extern int spot_(integer *komp, integer *n, doublereal *sinth, doublereal *costh, doublereal *sinfi, doublereal *cosfi, doublereal *temf);
extern int square_(doublereal *obs, integer *nobs, integer *ml, doublereal *out, doublereal *sd, doublereal *xlamda, doublereal *d__, doublereal *cn, doublereal *cnn, doublereal *cnc, doublereal *clc, doublereal *ss, doublereal *cl, integer *ll, integer *mm);
extern int surfas_(doublereal *rmass, doublereal *potent, integer *n, integer *n1, integer *komp, doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *fr1, doublereal *fr2, doublereal *hld, doublereal *ff, doublereal *d__, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *grv1, doublereal *grv2, doublereal *xx1, doublereal *yy1, doublereal *zz1, doublereal *xx2, doublereal *yy2, doublereal *zz2, doublereal *csbt1, doublereal *csbt2, doublereal *glump1, doublereal *glump2, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, doublereal *grexp);
extern int volume_(doublereal *v, doublereal *q, doublereal *p, doublereal *d__, doublereal *ff, integer *n, integer *n1, integer *komp, doublereal *rv, doublereal *grx, doublereal *gry, doublereal *grz, doublereal *rvq, doublereal *grxq, doublereal *gryq, doublereal *grzq, integer *mmsave, doublereal *fr1, doublereal *fr2, doublereal *hld, doublereal *snth, doublereal *csth, doublereal *snfi, doublereal *csfi, doublereal *summ, doublereal *sm, doublereal *grv1, doublereal *grv2, doublereal *xx1, doublereal *yy1, doublereal *zz1, doublereal *xx2, doublereal *yy2, doublereal *zz2, doublereal *csbt1, doublereal *csbt2, doublereal *glump1, doublereal *glump2, doublereal *gmag1, doublereal *gmag2, doublereal *glog1, doublereal *glog2, doublereal *grexp, integer *ifc);
extern int wrdata_(doublereal *hjd, doublereal *phas, doublereal *yskp, doublereal *zskp, doublereal *htt, doublereal *cool, doublereal *total, doublereal *tot, doublereal *d__, doublereal *smagg, doublereal *vsum1, doublereal *vsum2, doublereal *vra1, doublereal *vra2, doublereal *vkm1, doublereal *vkm2, doublereal *delv1, doublereal *delwl1, doublereal *wl1, doublereal *fbin1, doublereal *resf1, doublereal *delv2, doublereal *delwl2, doublereal *wl2, doublereal *fbin2, doublereal *resf2, doublereal *rv, doublereal *rvq, integer *mmsave, integer *ll1, integer *lll1, integer *llll1, integer *ll2, integer *lll2, integer *llll2);
extern int wrdci_(char *fn, doublereal *del, integer *kep, integer *ifder, integer *ifm, integer *ifr, doublereal *xlamda, integer *kspa, integer *nspa, integer *kspb, integer *nspb, integer *ifvc1, integer *ifvc2, integer *nlc, integer *k0, integer *kdisk, integer *isym, integer *nppl, integer *nref, integer *mref, integer *ifsmv1, integer *ifsmv2, integer *icor1, integer *icor2, integer *ld, integer *jdphs, doublereal *hjd0, doublereal *period, doublereal *dpdt, doublereal *pshift, integer *mode, integer *ipb, integer *ifat1, integer *ifat2, integer *n1, integer *n2, integer *n1l, integer *n2l, doublereal *perr0, doublereal *dperdt, doublereal *the, doublereal *vunit, doublereal *e, doublereal *a, doublereal *f1, doublereal *f2, doublereal *vga, doublereal *xincl, doublereal *gr1, doublereal *gr2, doublereal *abunin, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *phsv, doublereal *pcsv, doublereal *rm, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, integer *iband, doublereal *hla, doublereal *cla, doublereal *x1a, doublereal *x2a, doublereal *y1a, doublereal *y2a, doublereal *el3, doublereal *opsf, integer *noise, doublereal *sigma, doublereal *wla, integer *nsp1, doublereal *xlat1, doublereal *xlong1, doublereal *radsp1, doublereal *temsp1, integer *nsp2, doublereal *xlat2, doublereal *xlong2, doublereal *radsp2, doublereal *temsp2, integer *vertno, doublereal *indep, doublereal *dep, doublereal *weight, ftnlen fn_len);
extern int wrfoot_(integer *message, doublereal *f1, doublereal *f2, doublereal *po, doublereal *rm, doublereal *f, doublereal *dp, doublereal *e, doublereal *drdq, doublereal *dodq, integer *ii, integer *mode, integer *mpage);
extern int wrhead_(integer *ibef, integer *nref, integer *mref, integer *ifsmv1, integer *ifsmv2, integer *icor1, integer *icor2, integer *ld, integer *jdphs, doublereal *hjd0, doublereal *period, doublereal *dpdt, doublereal *pshift, doublereal *stdev, integer *noise, doublereal *seed, doublereal *hjdst, doublereal *hjdsp, doublereal *hjdin, doublereal *phstrt, doublereal *phstop, doublereal *phin, doublereal *phn, integer *mode, integer *ipb, integer *ifat1, integer *ifat2, integer *n1, integer *n2, doublereal *perr0, doublereal *dperdt, doublereal *the, doublereal *vunit, doublereal *vfac, doublereal *e, doublereal *a, doublereal *f1, doublereal *f2, doublereal *vga, doublereal *xincl, doublereal *gr1, doublereal *gr2, integer *nsp1, integer *nsp2, doublereal *abunin, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *phsv, doublereal *pcsv, doublereal *rm, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, integer *iband, doublereal *hlum, doublereal *clum, doublereal *xh, doublereal *xc, doublereal *yh, doublereal *yc, doublereal *el3, doublereal *opsf, doublereal *zero, doublereal *factor, doublereal *wl, doublereal *binwm1, doublereal *sc1, doublereal *sl1, doublereal *binwm2, doublereal *sc2, doublereal *sl2, doublereal *wll1, doublereal *ewid1, doublereal *depth1, doublereal *wll2, doublereal *ewid2, doublereal *depth2, integer *kks, doublereal *xlat, doublereal *xlong, doublereal *radsp, doublereal *temsp, doublereal *xcl, doublereal *ycl, doublereal *zcl, doublereal *rcl, doublereal *op1, doublereal *fcl, doublereal *edens, doublereal *xmue, doublereal *encl, doublereal *dens, integer *ns1, doublereal *sms1, doublereal *sr1, doublereal *bolm1, doublereal *xlg1, integer *ns2, doublereal *sms2, doublereal *sr2, doublereal *bolm2, doublereal *xlg2, integer *mmsave, doublereal *sbrh, doublereal *sbrc, doublereal *sm1, doublereal *sm2, doublereal *phperi, doublereal *pconsc, doublereal *pconic, doublereal *dif1, doublereal *abunir, doublereal *abun, integer *mod);
extern int wrlci_(char *fn, integer *mpage, integer *nref, integer *mref, integer *ifsmv1, integer *ifsmv2, integer *icor1, integer *icor2, integer *ld, integer *jdphs, doublereal *hjd0, doublereal *period, doublereal *dpdt, doublereal *pshift, doublereal *stddev, integer *noise, doublereal *seed, doublereal *jdstrt, doublereal *jdend, doublereal *jdinc, doublereal *phstrt, doublereal *phend, doublereal *phinc, doublereal *phnorm, integer *mode, integer *ipb, integer *ifat1, integer *ifat2, integer *n1, integer *n2, doublereal *perr0, doublereal *dperdt, doublereal *the, doublereal *vunit, doublereal *e, doublereal *sma, doublereal *f1, doublereal *f2, doublereal *vga, doublereal *xincl, doublereal *gr1, doublereal *gr2, doublereal *abunin, doublereal *tavh, doublereal *tavc, doublereal *alb1, doublereal *alb2, doublereal *phsv, doublereal *pcsv, doublereal *rm, doublereal *xbol1, doublereal *xbol2, doublereal *ybol1, doublereal *ybol2, integer *iband, doublereal *hla, doublereal *cla, doublereal *x1a, doublereal *x2a, doublereal *y1a, doublereal *y2a, doublereal *el3, doublereal *opsf, doublereal *mzero, doublereal *factor, doublereal *wla, integer *nsp1, doublereal *xlat1, doublereal *xlong1, doublereal *radsp1, doublereal *temsp1, integer *nsp2, doublereal *xlat2, doublereal *xlong2, doublereal *radsp2, doublereal *temsp2, ftnlen fn_len);

#endif
