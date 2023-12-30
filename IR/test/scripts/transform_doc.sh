XMLDIR="/home/sergey/MAI/maga/Inf_search/non_comm_use.O-Z.xml/"
# XMLDIR="/media/sergey/Transcend/Serega/non_comm_use.I-N.xml/"
XSLFILE="/home/sergey/MAI/maga/Inf_search/test/xsl_transform_documents_v3.xsl"
RESFILE="/home/sergey/MAI/maga/Inf_search/test/res_20210418.xml"
LOGFILE="/home/sergey/MAI/maga/Inf_search/test/res_20210418.log"
NXMLS=($(find $XMLDIR -name "*.nxml"));
Total_loop_num=${#NXMLS[*]};
echo $Total_loop_num;
# echo "<Articles>" > $RESFILE
for((i=0;i<$Total_loop_num;i++))
do
   echo -ne "Progress: $[ ($i+1)*100/$Total_loop_num ]%\r";
   f=${NXMLS[$i]};
   #echo $f; 
   xsltproc "$XSLFILE" "$f" 2>>$LOGFILE | tail -n+2 | head -n -1 >> $RESFILE
done;
echo "</Articles>" >> $RESFILE
echo "Loop finished.";
