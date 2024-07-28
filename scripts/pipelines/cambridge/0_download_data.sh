export dataset=datasets/cambridge
export scenes=( "KingsCollege" "OldHospital" "StMarysChurch" "ShopFacade" "GreatCourt" )
export IDs=( "251342" "251340" "251294" "251336" "251291" )
for i in "${!scenes[@]}"; do
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/${IDs[i]}/${scenes[i]}.zip -P $dataset \
&& unzip $dataset/${scenes[i]}.zip -d $dataset && rm $dataset/${scenes[i]}.zip; done

gdown -O datasets/test_folder/ 1esqzZ1zEQlzZVic-H32V6kkZvc4NeS15