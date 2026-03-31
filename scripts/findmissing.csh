#!/bin/csh -f
# Identify if any model files are not listed in models.tab
# 
foreach z ( 1 )
    cd ../pdrtpy/models/wolfirekaufman/version2020/constant_density/z=${z}
    foreach dir  ( losangle=* )
        cd $dir
        foreach file ( *sm.fits )
            set basefile = `basename $file`
            set search = ${basefile:r}
            #echo "grep -L $search models.tab"
            set x=`grep -L $search models.tab`
            #if ( "$x" == "models.tab" ) then
            if ( $? == 1 ) then
                echo $dir/models.tab missing $file
            endif
            #echo $search
        end
        cd ..
    end
end

