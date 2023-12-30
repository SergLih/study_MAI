#! /bin/bash
python2 conv.py house.png in_ol.data
nw=$((367/$1))
nh=$((376/$2))
printf "in_ol.data\nout_ol.data\n$nw $nh\n" > params.txt	
echo "New size: $nw x $nh"
./a.out < params.txt
python2 conv.py out_ol.data house_out.png
eog house_out.png
