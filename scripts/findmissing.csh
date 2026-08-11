#!/bin/csh -f
# Identify if any model files are not listed in models.tab
#
set dir=../pdrtpy/models/wolfirekaufman/version2020/constant_density/
set tab=models.tab
foreach arg (${argv})
    set ${arg}
    #echo "set ${arg}"
end
#foreach zed ( 1 )
    cd ${dir}/${z}
    foreach angle ( losangle=* )
        cd $angle
        #echo doing $dir/$z/$angle
        foreach file ( *sm.fits )
            set basefile = `basename $file`
            set search = ${basefile:r}
            set x=`grep -L $search ${tab}`
            if ( $? == 1 ) then
                echo ${dir}/${angle}/${tab} missing $file
            endif
            #echo $search
        end
        cd ..
    end
#end
