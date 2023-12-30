#!/bin/bash
python gen_tests_diff.py
diff $(cat opts.txt) 1.txt 2.txt > res.txt
./diff $(cat opts.txt) 1.txt 2.txt > my_res.txt
diff res.txt my_res.txt
#if $(diff res.txt my_res.txt)=''; then
#    echo 'ok';
#else
#    echo 'no';
#fi
