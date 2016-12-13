#!/bin/bash

#module load bedtools/2.21.0

thres=5 # threshold on -log10pvalue
mean_or_max=mean # take mean or max of -log10pvalue in each REsite

dataDIR=/mnt/lab_data/kundaje/projects/skin/data/peaks/
PROJDIR=/srv/scratch/mtaranov/peaks_at_promoters/
datatypes=(tf histone atac)
tf=(CTCF LSD1 PolII TP63)
histone=(H3K27ac H3K27me3 H3K4me1 H3K4me3 H3K9ac)
atac=(atac)
PromoterHindIII=/srv/scratch/mtaranov/HindIII/PromoterCapture_Digest_Human_HindIII_baits_ID.bed


output_folder=$PROJDIR/output_thres${thres}_${mean_or_max}
temp_folder=$output_folder/temp
mkdir -p $output_folder $temp_folder

for (( i=0; i<${#datatypes[@]}; i++ )); do
       datatype_all=${datatypes[$i]}
       if [[ $datatype_all == "tf" ]]; then
           sub_folder=""
       else
           sub_folder="narrow"
        fi
       
       eval datatype=\( \${${datatype_all}[@]} \) # iterate over a array using inderect reference
       for (( j=0; j<${#datatype[@]}; j++ )); do
           data=${datatype[$j]}
 
        if [[ $datatype_all == "atac" ]]; then
        narrowPeakFile=$dataDIR/$datatype_all/$sub_folder/khavari_hg19_primarykeratinocyte_all_ATAC_merged_r1.bed_p1e-2_peaks.narrowPeak
        elif [[ $data == "H3K27ac" ]]; then       
        narrowPeakFile=$dataDIR/$datatype_all/$sub_folder/khavari_hg19_primarykeratinocyte_d0_${data}_chipseq_r2_p1e-2_peaks.narrowPeak # rep 2
        else
        narrowPeakFile=$dataDIR/$datatype_all/$sub_folder/khavari_hg19_primarykeratinocyte_d0_${data}_chipseq_r1_p1e-2_peaks.narrowPeak
        fi 
        temp=$temp_folder/temp_${data}
        output=$output_folder/${data}

 
cat $narrowPeakFile | awk '{if ($8 > thres) print $1, $2, $3, $8}' thres=$thres | /usr/bin/perl -p -e 's/ /\t/g' > $temp

if [[ $mean_or_max == "mean" ]]; then
bedtools intersect -wao -a $PromoterHindIII -b $temp |awk '{if ($7 != -1) print $1, $2, $3, $9}' | sort -k 1,1 | gawk '($1==key1 && $2==key2) { sum+=$4; n++} ($1 != key1 || $2 != key2) {if (NR>1){print key1, key2, key3, sum/n} key1=$1; key2=$2; key3=$3; sum=$4; n=1}' >  $output

elif [[ $mean_or_max == "max" ]]; then
bedtools intersect -wao -a $PromoterHindIII -b $temp |awk '{if ($7 != -1) print $1, $2, $3, $9}' | sort -k 1,1 | gawk '($1==key1 && $2==key2) { if(($4)>max)  max=($4)} ($1 != key1 || $2 != key2) {if (NR>1){print key1, key2, key3, max} key1=$1; key2=$2; key3=$3; max=$4}' >  $output
fi

done
done

#rm  -r $temp_folder
exit
