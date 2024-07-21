export dataset=data/7scenes
#Download the SIFT SfM models and DenseVLAD image pairs, courtesy of Torsten Sattler:
function download {
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
unzip $2 -d $dataset && rm $2;
}
echo https://docs.google.com/uc?export=download&id=1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb
download 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb $dataset/7scenes_sfm_triangulated.zip
download 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 $dataset/7scenes_densevlad_retrieval_top_10.zip
