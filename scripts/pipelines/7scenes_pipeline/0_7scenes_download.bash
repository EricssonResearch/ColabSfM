export dataset="datasets/7scenes"
for scene in chess fire heads office pumpkin redkitchen stairs; \
do wget "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/${scene}.zip" -P ${dataset} \
&& unzip ${dataset}/${scene}.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done

sattler_sfm=7scenes_sfm_triangulated.zip
gdown -O $dataset/$sattler_sfm 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb && unzip ${dataset}/$sattler_sfm
gdown -O $dataset/7scenes_densevlad_retrieval_top_10.zip 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1
