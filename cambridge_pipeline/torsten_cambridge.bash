export fileid=1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15
export filename=datasets/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$fileid" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$fileid" -O $filename && rm -rf /tmp/cookies.txt
unzip $filename -d datasets/cambridge/