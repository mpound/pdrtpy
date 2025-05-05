pro aeinconvert
; program to convert A data from Roueff Abgrail et al. 2019 to format
; to run in pdrbourlot/pdr_hh
; line_h2.dat Roeuff data from PDR7_240919_r2131/data
;aquadh2.dat is from pdrbourlot/hh/data new file is aquadh2.dat_roueff 
  readcol,'line_h2.dat',ndum,nudum,nldum,e,aij,quant,vu,ju,vl,jl,format='I,I,I,F,F,A,I,I,I,I'
 print,aij[0],vu[0],ju[0],vl[0],jl[0]
;   n     nu     nl                E(K)          Aij(s-1)             quant:     vu      Ju      vl      Jl  info:      Description  Lambda (micrometres)

   readcol,'aquadh2.dat',vuo,vlo,juo,S,Q,O
;                   S              Q              O
;   0   0   2   0.294929E-10   0.000000E+00   0.000000E+00
   ilines = n_elements(vuo)
   jlo = juo
   print,ilines
    S = vuo
    Q = S
    O = S
    for il=0,n_elements(vuo)-1 DO BEGIN
;   for il=0,2 DO BEGIN
      isnew = where(vuo[il] EQ  vu AND vlo[il] EQ vl AND juo[il] EQ ju AND juo[il]-2 EQ jl)
      iqnew = where(vuo[il] EQ  vu AND vlo[il] EQ vl AND juo[il] EQ ju AND juo[il]-0 EQ jl)
      ionew = where(vuo[il] EQ  vu AND vlo[il] EQ vl AND juo[il] EQ ju AND juo[il]+2 EQ jl)
      S[il] = 0.0
      Q[il] = 0.0
      O[il] = 0.0
      IF(isnew ne -1)THEN S[il] = aij(isnew)
      IF(iqnew ne -1)THEN Q[il] = aij(iqnew)
      IF(ionew ne -1)THEN O[il] = aij(ionew)
;      print,vuo[il],vlo[il],juo[il],S,Q,O
   endfor
    help,S[0],Q[0]
    forprint,vuo,vlo,juo,S,Q,O,FORMAT="(I4,I4,I4,E15.6,E15.6,E15.6)",textout=3
;    forprint,vuo,vlo,juo,S,Q,O,textout=3
end
